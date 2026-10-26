//! Child-process supervision with pause (OS suspend), cancel and stderr capture.

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};

/// Returned when a job is cancelled.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Shared pause/cancel state for one job and the processes it runs.
#[derive(Default)]
pub struct Control {
    cancelled: AtomicBool,
    paused: Mutex<bool>,
    resumed: Condvar,
    children: Mutex<Vec<Arc<Mutex<Child>>>>,
}

impl Control {
    pub fn new() -> Arc<Control> {
        Arc::new(Control::default())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        *self.paused.lock().unwrap()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        for c in self.children.lock().unwrap().iter() {
            let mut c = c.lock().unwrap();
            let _ = os::resume(&c);
            let _ = c.kill();
        }
        *self.paused.lock().unwrap() = false;
        self.resumed.notify_all();
    }

    pub fn pause(&self) {
        *self.paused.lock().unwrap() = true;
        for c in self.children.lock().unwrap().iter() {
            let _ = os::suspend(&c.lock().unwrap());
        }
    }

    pub fn resume(&self) {
        for c in self.children.lock().unwrap().iter() {
            let _ = os::resume(&c.lock().unwrap());
        }
        *self.paused.lock().unwrap() = false;
        self.resumed.notify_all();
    }

    /// Blocks the calling worker thread while paused. Returns `Err(Cancelled)` if cancelled.
    pub fn checkpoint(&self) -> Result<()> {
        let mut p = self.paused.lock().unwrap();
        while *p && !self.is_cancelled() {
            p = self.resumed.wait(p).unwrap();
        }
        drop(p);
        if self.is_cancelled() { Err(Cancelled.into()) } else { Ok(()) }
    }

    fn register(&self, child: Arc<Mutex<Child>>) {
        if self.is_paused() {
            let _ = os::suspend(&child.lock().unwrap());
        }
        self.children.lock().unwrap().push(child);
    }

    fn unregister(&self, child: &Arc<Mutex<Child>>) {
        self.children.lock().unwrap().retain(|c| !Arc::ptr_eq(c, child));
    }
}

const TAIL_LINES: usize = 200;

/// A supervised child process.
pub struct Proc {
    child: Arc<Mutex<Child>>,
    pub label: String,
    tail: Arc<Mutex<VecDeque<String>>>,
    stderr_thread: Option<JoinHandle<()>>,
    control: Arc<Control>,
}

pub type LineHandler = Box<dyn FnMut(&str) + Send>;

pub struct Spawned {
    pub proc: Proc,
    pub stdin: Option<ChildStdin>,
    pub stdout: Option<ChildStdout>,
}

impl Proc {
    pub fn spawn(program: &Path, args: &[String], cwd: &Path, label: &str, pipe_stdin: bool, pipe_stdout: bool, mut on_line: Option<LineHandler>, control: &Arc<Control>) -> Result<Spawned> {
        control.checkpoint()?;
        let mut cmd = Command::new(program);
        cmd.args(args).current_dir(cwd).stderr(Stdio::piped());
        cmd.stdin(if pipe_stdin { Stdio::piped() } else { Stdio::null() });
        cmd.stdout(if pipe_stdout { Stdio::piped() } else { Stdio::null() });
        os::no_window(&mut cmd);
        let mut child = cmd.spawn().with_context(|| format!("starting {}", program.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail2 = tail.clone();
        let stderr_thread = stderr.map(|mut s| {
            std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                let mut line = Vec::new();
                let mut flush = |line: &mut Vec<u8>| {
                    if line.is_empty() {
                        return;
                    }
                    let text = String::from_utf8_lossy(line).trim().to_string();
                    line.clear();
                    if text.is_empty() {
                        return;
                    }
                    if let Some(h) = on_line.as_mut() {
                        h(&text);
                    }
                    // Progress lines are frequent and carry no diagnostics.
                    if !text.starts_with("frame=") {
                        let mut t = tail2.lock().unwrap();
                        if t.len() == TAIL_LINES {
                            t.pop_front();
                        }
                        t.push_back(text);
                    }
                };
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    for &b in &buf[..n] {
                        if b == b'\n' || b == b'\r' {
                            flush(&mut line);
                        } else {
                            line.push(b);
                        }
                    }
                }
                flush(&mut line);
            })
        });
        let child = Arc::new(Mutex::new(child));
        control.register(child.clone());
        Ok(Spawned { proc: Proc { child, label: label.into(), tail, stderr_thread, control: control.clone() }, stdin, stdout })
    }

    /// Waits for exit without holding the child lock, so pause/cancel stay responsive.
    pub fn wait(mut self) -> Result<(ExitStatus, Vec<String>)> {
        let status = loop {
            if let Some(s) = self.child.lock().unwrap().try_wait()? {
                break s;
            }
            std::thread::sleep(Duration::from_millis(15));
        };
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
        self.control.unregister(&self.child);
        let tail = self.tail.lock().unwrap().iter().cloned().collect();
        Ok((status, tail))
    }

    pub fn kill(&self) {
        let mut c = self.child.lock().unwrap();
        let _ = os::resume(&c);
        let _ = c.kill();
    }

    pub fn tail(&self) -> Vec<String> {
        self.tail.lock().unwrap().iter().cloned().collect()
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        // A Proc dropped without `wait` belongs to a failed pipeline; make sure it does not linger.
        if self.stderr_thread.is_some() {
            self.kill();
            let _ = self.child.lock().unwrap().wait();
            self.control.unregister(&self.child);
        }
    }
}

#[cfg(windows)]
mod os {
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};

    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};

    type NtProcFn = unsafe extern "system" fn(*mut core::ffi::c_void) -> i32;

    fn ntdll(name: &'static [u8]) -> Option<NtProcFn> {
        // SAFETY: ntdll is always loaded; the looked-up exports have the declared signature.
        unsafe {
            let module = GetModuleHandleA(c"ntdll.dll".as_ptr() as _);
            if module.is_null() {
                return None;
            }
            GetProcAddress(module, name.as_ptr()).map(|f| std::mem::transmute::<_, NtProcFn>(f))
        }
    }

    fn call(child: &Child, name: &'static [u8]) -> io::Result<()> {
        let f = ntdll(name).ok_or_else(|| io::Error::other("ntdll export missing"))?;
        // SAFETY: the handle belongs to a live child process we own.
        let status = unsafe { f(child.as_raw_handle() as _) };
        if status < 0 { Err(io::Error::other(format!("NTSTATUS 0x{status:08x}"))) } else { Ok(()) }
    }

    pub fn suspend(child: &Child) -> io::Result<()> {
        call(child, b"NtSuspendProcess\0")
    }

    pub fn resume(child: &Child) -> io::Result<()> {
        call(child, b"NtResumeProcess\0")
    }

    pub fn no_window(cmd: &mut Command) {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

#[cfg(unix)]
mod os {
    use std::io;
    use std::process::{Child, Command};

    fn signal(child: &Child, sig: i32) -> io::Result<()> {
        // SAFETY: plain kill(2) on our own child's pid.
        if unsafe { libc::kill(child.id() as i32, sig) } == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
    }

    pub fn suspend(child: &Child) -> io::Result<()> {
        signal(child, libc::SIGSTOP)
    }

    pub fn resume(child: &Child) -> io::Result<()> {
        signal(child, libc::SIGCONT)
    }

    pub fn no_window(_cmd: &mut Command) {}
}

/// Creates a `Command` that does not open a console window on Windows.
pub fn command(program: &Path) -> Command {
    let mut c = Command::new(program);
    os::no_window(&mut c);
    c
}

//! Export queue: N workers run jobs by priority (VeryLow 1 .. VeryHigh 5, Normal 3), oldest first.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use prism_core::models::Catalog;
use prism_core::plan::JobPlan;
use prism_core::progress::Progress;

use crate::ffmpeg::Tools;
use crate::process::Control;
use crate::runner::{self, Event, RunReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Waiting,
    Processing,
    Paused,
    Done,
    Error,
    Cancelled,
}

impl JobState {
    pub fn is_finished(self) -> bool {
        matches!(self, JobState::Done | JobState::Error | JobState::Cancelled)
    }
}

pub const PRIORITY_NORMAL: u8 = 3;

#[derive(Clone)]
pub struct JobView {
    pub id: u64,
    pub name: String,
    pub output: std::path::PathBuf,
    pub state: JobState,
    pub priority: u8,
    pub progress: Progress,
    pub pass_label: String,
    pub message: String,
    pub error: Option<String>,
    pub report: Option<RunReport>,
    pub log: Vec<String>,
}

struct Job {
    view: JobView,
    plan: Arc<JobPlan>,
    control: Arc<Control>,
    seq: u64,
}

struct State {
    jobs: Vec<Job>,
    next_id: u64,
    workers_wanted: usize,
    running: usize,
    shutdown: bool,
}

pub struct Queue {
    state: Arc<(Mutex<State>, Condvar)>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    tools: Tools,
    catalog: Arc<Catalog>,
}

impl Queue {
    pub fn new(tools: Tools, catalog: Arc<Catalog>, workers: usize) -> Arc<Queue> {
        let q = Arc::new(Queue {
            state: Arc::new((Mutex::new(State { jobs: Vec::new(), next_id: 1, workers_wanted: workers.max(1), running: 0, shutdown: false }), Condvar::new())),
            workers: Mutex::new(Vec::new()),
            tools,
            catalog,
        });
        q.spawn_workers(workers.max(1));
        q
    }

    fn spawn_workers(self: &Arc<Self>, n: usize) {
        let mut ws = self.workers.lock().unwrap();
        for _ in 0..n {
            let state = self.state.clone();
            let tools = self.tools.clone();
            let catalog = self.catalog.clone();
            ws.push(std::thread::spawn(move || worker(state, tools, catalog)));
        }
    }

    pub fn submit(&self, name: &str, plan: JobPlan, priority: u8) -> u64 {
        let (lock, cv) = &*self.state;
        let mut s = lock.lock().unwrap();
        let id = s.next_id;
        s.next_id += 1;
        let view = JobView {
            id,
            name: name.into(),
            output: plan.output.clone(),
            state: JobState::Waiting,
            priority: priority.clamp(1, 5),
            progress: Progress { total: plan.frames_out, ..Default::default() },
            pass_label: String::new(),
            message: "Waiting".into(),
            error: None,
            report: None,
            log: plan.warnings.clone(),
        };
        s.jobs.push(Job { view, plan: Arc::new(plan), control: Control::new(), seq: id });
        cv.notify_all();
        id
    }

    pub fn jobs(&self) -> Vec<JobView> {
        self.state.0.lock().unwrap().jobs.iter().map(|j| j.view.clone()).collect()
    }

    pub fn get(&self, id: u64) -> Option<JobView> {
        self.state.0.lock().unwrap().jobs.iter().find(|j| j.view.id == id).map(|j| j.view.clone())
    }

    fn with_job<R>(&self, id: u64, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
        let (lock, cv) = &*self.state;
        let mut s = lock.lock().unwrap();
        let r = s.jobs.iter_mut().find(|j| j.view.id == id).map(f);
        cv.notify_all();
        r
    }

    pub fn pause(&self, id: u64) {
        self.with_job(id, |j| {
            if j.view.state == JobState::Processing {
                j.control.pause();
                j.view.state = JobState::Paused;
                j.view.message = "Paused".into();
            }
        });
    }

    pub fn resume(&self, id: u64) {
        self.with_job(id, |j| {
            if j.view.state == JobState::Paused {
                j.control.resume();
                j.view.state = JobState::Processing;
                j.view.message = "Resumed".into();
            }
        });
    }

    pub fn cancel(&self, id: u64) {
        self.with_job(id, |j| match j.view.state {
            JobState::Waiting => {
                j.view.state = JobState::Cancelled;
                j.view.message = "Cancelled".into();
            }
            JobState::Processing | JobState::Paused => j.control.cancel(),
            _ => {}
        });
    }

    pub fn set_priority(&self, id: u64, priority: u8) {
        self.with_job(id, |j| j.view.priority = priority.clamp(1, 5));
    }

    /// Removes finished jobs from the list.
    pub fn clear_finished(&self) {
        self.state.0.lock().unwrap().jobs.retain(|j| !j.view.state.is_finished());
    }

    pub fn set_workers(self: &Arc<Self>, n: usize) {
        let n = n.max(1);
        let current = {
            let mut s = self.state.0.lock().unwrap();
            let cur = s.workers_wanted;
            s.workers_wanted = n;
            cur
        };
        if n > current {
            self.spawn_workers(n - current);
        }
        self.state.1.notify_all();
    }

    pub fn is_idle(&self) -> bool {
        let s = self.state.0.lock().unwrap();
        s.running == 0 && !s.jobs.iter().any(|j| j.view.state == JobState::Waiting)
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        {
            let mut s = self.state.0.lock().unwrap();
            s.shutdown = true;
            for j in &s.jobs {
                j.control.cancel();
            }
        }
        self.state.1.notify_all();
    }
}

fn worker(state: Arc<(Mutex<State>, Condvar)>, tools: Tools, catalog: Arc<Catalog>) {
    let (lock, cv) = &*state;
    loop {
        let (id, plan, control) = {
            let mut s = lock.lock().unwrap();
            loop {
                if s.shutdown {
                    return;
                }
                // Shrinking the pool: surplus idle workers exit.
                let alive = s.workers_wanted;
                if s.running >= alive {
                    s = cv.wait(s).unwrap();
                    continue;
                }
                let pick = s
                    .jobs
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| j.view.state == JobState::Waiting)
                    .max_by(|a, b| a.1.view.priority.cmp(&b.1.view.priority).then(b.1.seq.cmp(&a.1.seq)))
                    .map(|(i, _)| i);
                match pick {
                    Some(i) => {
                        s.running += 1;
                        let j = &mut s.jobs[i];
                        j.view.state = JobState::Processing;
                        j.view.message = "Starting".into();
                        break (j.view.id, j.plan.clone(), j.control.clone());
                    }
                    None => s = cv.wait(s).unwrap(),
                }
            }
        };
        let state2 = state.clone();
        let on_event = move |ev: Event| {
            let mut s = state2.0.lock().unwrap();
            if let Some(j) = s.jobs.iter_mut().find(|j| j.view.id == id) {
                match ev {
                    Event::PassStarted { index, count, label } => {
                        j.view.pass_label = if count > 1 { format!("{label} ({}/{count})", index + 1) } else { label };
                        j.view.message = j.view.pass_label.clone();
                    }
                    Event::Progress(p) => {
                        j.view.progress = p;
                        if j.view.state == JobState::Processing {
                            j.view.message = if p.finalizing { "Finalizing...".into() } else { j.view.pass_label.clone() };
                        }
                    }
                    Event::Info(m) | Event::Warning(m) => j.view.log.push(m),
                }
            }
        };
        let result = runner::run_plan(&plan, &tools, &catalog, &control, &on_event);
        let mut s = lock.lock().unwrap();
        s.running -= 1;
        if let Some(j) = s.jobs.iter_mut().find(|j| j.view.id == id) {
            match result {
                Ok(report) => {
                    j.view.state = JobState::Done;
                    j.view.message = format!("Done in {}", prism_core::progress::format_duration(report.elapsed_secs));
                    j.view.progress.percent = 100;
                    j.view.report = Some(report);
                }
                Err(e) if runner::is_cancelled(&e) => {
                    j.view.state = JobState::Cancelled;
                    j.view.message = "Cancelled".into();
                }
                Err(e) => {
                    j.view.state = JobState::Error;
                    j.view.message = runner::summarize_error(&e);
                    j.view.error = Some(format!("{e:#}"));
                }
            }
        }
        cv.notify_all();
    }
}

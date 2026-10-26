//! Queue pause/resume/cancel against real FFmpeg processes (skipped when FFmpeg is missing).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use prism_core::models::Catalog;
use prism_core::plan::{PlanContext, plan};
use prism_core::JobSettings;
use prism_engine::Tools;
use prism_engine::process::command;
use prism_engine::queue::{JobState, Queue};

fn clip(tools: &Tools, dir: &Path) -> PathBuf {
    let path = dir.join("clip.mp4");
    let ok = command(&tools.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", "8", "-c:v", "libx264", "-preset", "ultrafast"])
        .arg(&path)
        .status()
        .unwrap()
        .success();
    assert!(ok);
    path
}

fn wait_for(queue: &Queue, id: u64, pred: impl Fn(&prism_engine::queue::JobView) -> bool, timeout: Duration) -> prism_engine::queue::JobView {
    let start = Instant::now();
    loop {
        let v = queue.get(id).unwrap();
        if pred(&v) || start.elapsed() > timeout {
            return v;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn pause_freezes_progress_and_cancel_removes_output() {
    let Ok(tools) = Tools::locate() else {
        eprintln!("skipping: FFmpeg not found");
        return;
    };
    let dir = std::env::temp_dir().join(format!("prism-queue-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = clip(&tools, &dir);
    let info = tools.probe(&input).unwrap();
    let catalog = Arc::new(Catalog::load(Path::new("no-models")));
    let mut s = JobSettings::default();
    // Slow enough to observe: motion interpolation to 4x the frame rate.
    s.interpolation.enabled = true;
    s.interpolation.output_fps = Some(120.0);
    s.export.output_dir = Some(dir.clone());
    let exists = |p: &Path| p.exists();
    let mk_plan = |name: &str| {
        let ctx = PlanContext { catalog: &catalog, temp_dir: dir.join(name), encoders_available: None, preview: None, exists: &exists };
        let mut s2 = s.clone();
        s2.export.output_path = Some(dir.join(format!("{name}.mp4")));
        plan(&info, &s2, &ctx).unwrap()
    };
    let queue = Queue::new(tools.clone(), catalog.clone(), 1);

    // Pause and resume.
    let id = queue.submit("pause", mk_plan("pause"), 3);
    let v = wait_for(&queue, id, |v| v.progress.frame > 20, Duration::from_secs(60));
    assert_eq!(v.state, JobState::Processing, "job should be running: {:?}", v.message);
    queue.pause(id);
    std::thread::sleep(Duration::from_millis(400));
    let frozen = queue.get(id).unwrap().progress.frame;
    std::thread::sleep(Duration::from_millis(1200));
    let still = queue.get(id).unwrap();
    assert_eq!(still.state, JobState::Paused);
    assert!(still.progress.frame <= frozen + 2, "progress advanced while paused: {frozen} -> {}", still.progress.frame);
    queue.resume(id);
    let done = wait_for(&queue, id, |v| v.state.is_finished(), Duration::from_secs(120));
    assert_eq!(done.state, JobState::Done, "{:?}", done.error);
    assert!(dir.join("pause.mp4").exists());

    // Cancel removes the partial output.
    let id = queue.submit("cancel", mk_plan("cancel"), 3);
    wait_for(&queue, id, |v| v.progress.frame > 20, Duration::from_secs(60));
    queue.cancel(id);
    let v = wait_for(&queue, id, |v| v.state.is_finished(), Duration::from_secs(30));
    assert_eq!(v.state, JobState::Cancelled);
    assert!(!dir.join("cancel.mp4").exists(), "partial output was left behind");
    let _ = std::fs::remove_dir_all(&dir);
}

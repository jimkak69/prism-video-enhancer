//! Line icons drawn with the painter, so the app ships no image assets.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2, vec2};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    Logo,
    Play,
    Pause,
    StepBack,
    StepForward,
    JumpIn,
    JumpOut,
    FullScreen,
    SingleView,
    SplitView,
    SideBySide,
    Presets,
    Gear,
    Panel,
    Crop,
    ChevronRight,
    ChevronDown,
    ChevronLeft,
    Close,
    Plus,
    Minus,
    Cube,
    Sources,
    Queue,
    Star,
    StarFilled,
    Lock,
    Folder,
    ExportOut,
    Carousel,
    List,
    Ellipsis,
    Check,
    MarkIn,
    MarkOut,
    ClearInOut,
    ZoomReset,
    Minimize,
    Maximize,
    Restore,
    Download,
    Stop,
    Warning,
    Film,
}

fn line(p: &Painter, a: Pos2, b: Pos2, s: Stroke) {
    p.line_segment([a, b], s);
}

fn poly(p: &Painter, pts: Vec<Pos2>, s: Stroke) {
    p.add(Shape::line(pts, s));
}

fn closed(p: &Painter, pts: Vec<Pos2>, s: Stroke) {
    p.add(Shape::closed_line(pts, s));
}

fn star_points(c: Pos2, r: f32) -> Vec<Pos2> {
    (0..10)
        .map(|i| {
            let a = -std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::PI / 5.0;
            let rr = if i % 2 == 0 { r } else { r * 0.45 };
            c + vec2(a.cos(), a.sin()) * rr
        })
        .collect()
}

/// Paints `icon` centred in `rect`, scaled to its smaller side.
pub fn paint(p: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let s = rect.width().min(rect.height());
    let c = rect.center();
    let u = s / 16.0;
    let at = |x: f32, y: f32| c + vec2((x - 8.0) * u, (y - 8.0) * u);
    let st = Stroke::new((1.3 * u).max(1.0), color);
    match icon {
        Icon::Logo => {
            let tri = vec![at(8.0, 1.5), at(15.0, 14.0), at(1.0, 14.0)];
            p.add(Shape::convex_polygon(tri, color, Stroke::NONE));
            p.add(Shape::convex_polygon(vec![at(8.0, 5.5), at(11.6, 12.0), at(4.4, 12.0)], Color32::from_rgb(0x21, 0x21, 0x21), Stroke::NONE));
            line(p, at(8.0, 5.5), at(8.0, 12.0), Stroke::new(1.2 * u, color));
        }
        Icon::Play => closed(p, vec![at(4.5, 2.5), at(13.0, 8.0), at(4.5, 13.5)], st),
        Icon::Pause => {
            p.rect_stroke(Rect::from_min_max(at(4.0, 3.0), at(6.5, 13.0)), 0.5, st, StrokeKind::Middle);
            p.rect_stroke(Rect::from_min_max(at(9.5, 3.0), at(12.0, 13.0)), 0.5, st, StrokeKind::Middle);
        }
        Icon::StepBack => {
            closed(p, vec![at(10.5, 3.5), at(4.0, 8.0), at(10.5, 12.5)], st);
            line(p, at(13.0, 3.0), at(13.0, 13.0), st);
        }
        Icon::StepForward => {
            closed(p, vec![at(5.5, 3.5), at(12.0, 8.0), at(5.5, 12.5)], st);
            line(p, at(3.0, 3.0), at(3.0, 13.0), st);
        }
        Icon::JumpIn => {
            line(p, at(3.0, 3.0), at(3.0, 13.0), st);
            closed(p, vec![at(13.0, 3.0), at(5.5, 8.0), at(13.0, 13.0)], st);
        }
        Icon::JumpOut => {
            line(p, at(13.0, 3.0), at(13.0, 13.0), st);
            closed(p, vec![at(3.0, 3.0), at(10.5, 8.0), at(3.0, 13.0)], st);
        }
        Icon::FullScreen => {
            p.rect_stroke(Rect::from_min_max(at(1.5, 3.0), at(14.5, 13.0)), 1.5 * u, st, StrokeKind::Middle);
            poly(p, vec![at(9.0, 5.5), at(12.0, 5.5), at(12.0, 8.5)], st);
            poly(p, vec![at(4.0, 7.5), at(4.0, 10.5), at(7.0, 10.5)], st);
        }
        Icon::SingleView => {
            p.rect_stroke(Rect::from_min_max(at(1.5, 3.5), at(14.5, 12.5)), 1.5 * u, st, StrokeKind::Middle);
        }
        Icon::SplitView => {
            p.rect_stroke(Rect::from_min_max(at(1.5, 3.5), at(14.5, 12.5)), 1.5 * u, st, StrokeKind::Middle);
            line(p, at(8.0, 1.5), at(8.0, 14.5), st);
        }
        Icon::SideBySide => {
            p.rect_stroke(Rect::from_min_max(at(0.8, 4.0), at(7.2, 12.0)), 1.2 * u, st, StrokeKind::Middle);
            p.rect_stroke(Rect::from_min_max(at(8.8, 4.0), at(15.2, 12.0)), 1.2 * u, st, StrokeKind::Middle);
        }
        Icon::Presets => {
            p.circle_stroke(at(6.0, 8.0), 4.5 * u, st);
            p.circle_stroke(at(10.0, 8.0), 4.5 * u, st);
        }
        Icon::Gear => {
            p.circle_stroke(c, 2.2 * u, st);
            p.circle_stroke(c, 5.0 * u, st);
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::FRAC_PI_4;
                let d = vec2(a.cos(), a.sin());
                line(p, c + d * 5.0 * u, c + d * 7.0 * u, Stroke::new(st.width * 1.6, color));
            }
        }
        Icon::Panel => {
            p.rect_stroke(Rect::from_min_max(at(1.5, 2.5), at(14.5, 13.5)), 1.5 * u, st, StrokeKind::Middle);
            line(p, at(9.5, 2.5), at(9.5, 13.5), st);
            line(p, at(11.3, 5.0), at(12.8, 5.0), st);
            line(p, at(11.3, 7.0), at(12.8, 7.0), st);
        }
        Icon::Crop => {
            poly(p, vec![at(4.5, 1.5), at(4.5, 11.5), at(14.5, 11.5)], st);
            poly(p, vec![at(1.5, 4.5), at(11.5, 4.5), at(11.5, 14.5)], st);
        }
        Icon::ChevronRight => poly(p, vec![at(6.0, 3.5), at(10.5, 8.0), at(6.0, 12.5)], st),
        Icon::ChevronLeft => poly(p, vec![at(10.0, 3.5), at(5.5, 8.0), at(10.0, 12.5)], st),
        Icon::ChevronDown => poly(p, vec![at(3.5, 6.0), at(8.0, 10.5), at(12.5, 6.0)], st),
        Icon::Close => {
            line(p, at(4.0, 4.0), at(12.0, 12.0), st);
            line(p, at(12.0, 4.0), at(4.0, 12.0), st);
        }
        Icon::Plus => {
            line(p, at(8.0, 2.5), at(8.0, 13.5), st);
            line(p, at(2.5, 8.0), at(13.5, 8.0), st);
        }
        Icon::Minus => line(p, at(3.0, 8.0), at(13.0, 8.0), st),
        Icon::Cube => {
            closed(p, vec![at(8.0, 1.5), at(14.0, 4.8), at(14.0, 11.2), at(8.0, 14.5), at(2.0, 11.2), at(2.0, 4.8)], st);
            poly(p, vec![at(2.0, 4.8), at(8.0, 8.0), at(14.0, 4.8)], st);
            line(p, at(8.0, 8.0), at(8.0, 14.5), st);
        }
        Icon::Sources => {
            line(p, at(8.0, 2.0), at(8.0, 10.0), st);
            poly(p, vec![at(4.5, 5.5), at(8.0, 2.0), at(11.5, 5.5)], st);
            for x in [2.0, 5.0, 8.0, 11.0, 14.0] {
                line(p, at(x - 0.6, 13.5), at(x + 0.6, 13.5), st);
            }
        }
        Icon::Queue => {
            closed(p, vec![at(8.0, 2.0), at(14.0, 5.5), at(8.0, 9.0), at(2.0, 5.5)], st);
            poly(p, vec![at(2.0, 8.5), at(8.0, 12.0), at(14.0, 8.5)], st);
            poly(p, vec![at(2.0, 11.0), at(8.0, 14.5), at(14.0, 11.0)], st);
        }
        Icon::Star => closed(p, star_points(at(8.0, 8.5), 6.8 * u), st),
        Icon::StarFilled => {
            let pts = star_points(at(8.0, 8.5), 6.8 * u);
            // Non-convex: fill as a fan of triangles around the centre.
            let centre = at(8.0, 8.5);
            for i in 0..pts.len() {
                p.add(Shape::convex_polygon(vec![centre, pts[i], pts[(i + 1) % pts.len()]], color, Stroke::NONE));
            }
        }
        Icon::Lock => {
            p.rect_stroke(Rect::from_min_max(at(3.0, 7.0), at(13.0, 14.0)), 1.5 * u, st, StrokeKind::Middle);
            poly(p, vec![at(5.5, 7.0), at(5.5, 4.5), at(6.5, 2.8), at(9.5, 2.8), at(10.5, 4.5), at(10.5, 7.0)], st);
        }
        Icon::Folder => closed(p, vec![at(1.5, 3.0), at(6.0, 3.0), at(7.5, 4.8), at(14.5, 4.8), at(14.5, 13.0), at(1.5, 13.0)], st),
        Icon::ExportOut => {
            poly(p, vec![at(6.0, 4.0), at(2.0, 4.0), at(2.0, 14.0), at(12.0, 14.0), at(12.0, 10.0)], st);
            line(p, at(7.0, 9.0), at(14.0, 2.0), st);
            poly(p, vec![at(9.5, 2.0), at(14.0, 2.0), at(14.0, 6.5)], st);
        }
        Icon::Carousel => {
            p.rect_stroke(Rect::from_min_max(at(4.5, 3.5), at(11.5, 12.5)), 1.0 * u, st, StrokeKind::Middle);
            line(p, at(1.5, 5.0), at(1.5, 11.0), st);
            line(p, at(14.5, 5.0), at(14.5, 11.0), st);
        }
        Icon::List => {
            for y in [4.0, 8.0, 12.0] {
                line(p, at(2.5, y), at(13.5, y), st);
            }
        }
        Icon::Ellipsis => {
            for x in [3.5, 8.0, 12.5] {
                p.circle_filled(at(x, 8.0), 1.2 * u, color);
            }
        }
        Icon::Check => poly(p, vec![at(3.0, 8.5), at(6.5, 12.0), at(13.0, 4.5)], Stroke::new(st.width * 1.3, color)),
        Icon::MarkIn => poly(p, vec![at(10.0, 2.5), at(6.0, 2.5), at(6.0, 13.5), at(10.0, 13.5)], st),
        Icon::MarkOut => poly(p, vec![at(6.0, 2.5), at(10.0, 2.5), at(10.0, 13.5), at(6.0, 13.5)], st),
        Icon::ClearInOut => {
            poly(p, vec![at(4.0, 2.5), at(1.5, 2.5), at(1.5, 13.5), at(4.0, 13.5)], st);
            poly(p, vec![at(12.0, 2.5), at(14.5, 2.5), at(14.5, 13.5), at(12.0, 13.5)], st);
            line(p, at(5.5, 5.5), at(10.5, 10.5), st);
            line(p, at(10.5, 5.5), at(5.5, 10.5), st);
        }
        Icon::ZoomReset => {
            poly(p, vec![at(4.0, 2.5), at(1.5, 2.5), at(1.5, 13.5), at(4.0, 13.5)], st);
            poly(p, vec![at(12.0, 2.5), at(14.5, 2.5), at(14.5, 13.5), at(12.0, 13.5)], st);
            line(p, at(4.5, 8.0), at(11.5, 8.0), st);
            poly(p, vec![at(6.5, 6.0), at(4.5, 8.0), at(6.5, 10.0)], st);
            poly(p, vec![at(9.5, 6.0), at(11.5, 8.0), at(9.5, 10.0)], st);
        }
        Icon::Minimize => line(p, at(3.0, 8.0), at(13.0, 8.0), Stroke::new(1.0, color)),
        Icon::Maximize => {
            p.rect_stroke(Rect::from_min_max(at(3.0, 3.0), at(13.0, 13.0)), 0.0, Stroke::new(1.0, color), StrokeKind::Middle);
        }
        Icon::Restore => {
            p.rect_stroke(Rect::from_min_max(at(3.0, 5.0), at(11.0, 13.0)), 0.0, Stroke::new(1.0, color), StrokeKind::Middle);
            poly(p, vec![at(5.0, 5.0), at(5.0, 3.0), at(13.0, 3.0), at(13.0, 11.0), at(11.0, 11.0)], Stroke::new(1.0, color));
        }
        Icon::Download => {
            line(p, at(8.0, 2.0), at(8.0, 10.5), st);
            poly(p, vec![at(4.5, 7.0), at(8.0, 10.5), at(11.5, 7.0)], st);
            line(p, at(2.5, 13.5), at(13.5, 13.5), st);
        }
        Icon::Stop => {
            p.rect_stroke(Rect::from_min_max(at(3.5, 3.5), at(12.5, 12.5)), 1.0 * u, st, StrokeKind::Middle);
        }
        Icon::Warning => {
            closed(p, vec![at(8.0, 2.0), at(14.5, 13.5), at(1.5, 13.5)], st);
            line(p, at(8.0, 6.0), at(8.0, 9.5), st);
            p.circle_filled(at(8.0, 11.5), 0.8 * u, color);
        }
        Icon::Film => {
            p.rect_stroke(Rect::from_min_max(at(2.0, 2.0), at(14.0, 14.0)), 1.5 * u, st, StrokeKind::Middle);
            line(p, at(5.0, 2.0), at(5.0, 14.0), st);
            line(p, at(11.0, 2.0), at(11.0, 14.0), st);
            line(p, at(2.0, 8.0), at(5.0, 8.0), st);
            line(p, at(11.0, 8.0), at(14.0, 8.0), st);
        }
    }
}

/// Paints an icon of `size` points centred at `centre`.
pub fn paint_at(p: &Painter, centre: Pos2, size: f32, icon: Icon, color: Color32) {
    paint(p, Rect::from_center_size(centre, Vec2::splat(size)), icon, color);
}

//! Minimal ONNX graph builder and protobuf encoder (just the messages a convolutional network needs).

use half::f16;

pub const FLOAT: i32 = 1;
pub const FLOAT16: i32 = 10;

#[derive(Default)]
struct Pb(Vec<u8>);

impl Pb {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }
    fn key(&mut self, field: u32, wire: u32) {
        self.varint(((field << 3) | wire) as u64);
    }
    fn int(&mut self, field: u32, v: i64) {
        self.key(field, 0);
        self.varint(v as u64);
    }
    fn float(&mut self, field: u32, v: f32) {
        self.key(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, field: u32, b: &[u8]) {
        self.key(field, 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    fn string(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }
    fn msg(&mut self, field: u32, m: Pb) {
        self.bytes(field, &m.0);
    }
}

pub enum Attr {
    Int(&'static str, i64),
    Ints(&'static str, Vec<i64>),
    Float(&'static str, f32),
    Str(&'static str, String),
}

struct Node {
    op: &'static str,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<Attr>,
}

struct Init {
    name: String,
    dims: Vec<i64>,
    dtype: i32,
    raw: Vec<u8>,
}

/// Dimension of a graph input/output: fixed or symbolic.
pub enum Dim {
    Fixed(i64),
    Named(&'static str),
}

pub struct Graph {
    nodes: Vec<Node>,
    inits: Vec<Init>,
    inputs: Vec<(String, i32, Vec<Dim>)>,
    outputs: Vec<(String, i32, Vec<Dim>)>,
    counter: usize,
    /// Element type of initializers created by [`Graph::weight`].
    pub weight_type: i32,
}

impl Graph {
    pub fn new(fp16: bool) -> Graph {
        Graph { nodes: Vec::new(), inits: Vec::new(), inputs: Vec::new(), outputs: Vec::new(), counter: 0, weight_type: if fp16 { FLOAT16 } else { FLOAT } }
    }

    pub fn input(&mut self, name: &str, dims: Vec<Dim>) -> String {
        self.inputs.push((name.into(), FLOAT, dims));
        name.into()
    }

    pub fn output(&mut self, value: &str, name: &str, dims: Vec<Dim>) {
        self.node_named("Identity", &[value], name, vec![]);
        self.outputs.push((name.into(), FLOAT, dims));
    }

    /// Adds a weight initializer in the graph's weight precision.
    pub fn weight(&mut self, prefix: &str, dims: &[usize], data: &[f32]) -> String {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "{prefix}: shape/data mismatch");
        let name = format!("{prefix}_{}", self.counter);
        self.counter += 1;
        let raw = if self.weight_type == FLOAT16 {
            data.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect()
        } else {
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        self.inits.push(Init { name: name.clone(), dims: dims.iter().map(|d| *d as i64).collect(), dtype: self.weight_type, raw });
        name
    }

    /// Adds an FP32 initializer regardless of the weight precision (e.g. `Resize` scales).
    pub fn const_f32(&mut self, prefix: &str, dims: &[usize], data: &[f32]) -> String {
        let name = format!("{prefix}_{}", self.counter);
        self.counter += 1;
        self.inits.push(Init { name: name.clone(), dims: dims.iter().map(|d| *d as i64).collect(), dtype: FLOAT, raw: data.iter().flat_map(|v| v.to_le_bytes()).collect() });
        name
    }

    pub fn node(&mut self, op: &'static str, inputs: &[&str], attrs: Vec<Attr>) -> String {
        let out = format!("{}_{}", op.to_ascii_lowercase(), self.counter);
        self.counter += 1;
        self.node_named(op, inputs, &out, attrs);
        out
    }

    fn node_named(&mut self, op: &'static str, inputs: &[&str], output: &str, attrs: Vec<Attr>) {
        self.nodes.push(Node { op, inputs: inputs.iter().map(|s| s.to_string()).collect(), outputs: vec![output.into()], attrs });
    }

    pub fn conv3x3(&mut self, x: &str, w: &str, b: &str) -> String {
        self.node("Conv", &[x, w, b], vec![Attr::Ints("kernel_shape", vec![3, 3]), Attr::Ints("pads", vec![1, 1, 1, 1]), Attr::Ints("strides", vec![1, 1])])
    }

    pub fn cast(&mut self, x: &str, to: i32) -> String {
        self.node("Cast", &[x], vec![Attr::Int("to", to as i64)])
    }

    pub fn encode(&self, opset: i64, doc: &str) -> Vec<u8> {
        let mut g = Pb::default();
        for n in &self.nodes {
            let mut m = Pb::default();
            for i in &n.inputs {
                m.string(1, i);
            }
            for o in &n.outputs {
                m.string(2, o);
            }
            m.string(4, n.op);
            for a in &n.attrs {
                let mut am = Pb::default();
                match a {
                    Attr::Int(name, v) => {
                        am.string(1, name);
                        am.int(3, *v);
                        am.int(20, 2);
                    }
                    Attr::Ints(name, v) => {
                        am.string(1, name);
                        for x in v {
                            am.int(8, *x);
                        }
                        am.int(20, 7);
                    }
                    Attr::Float(name, v) => {
                        am.string(1, name);
                        am.float(2, *v);
                        am.int(20, 1);
                    }
                    Attr::Str(name, v) => {
                        am.string(1, name);
                        am.bytes(4, v.as_bytes());
                        am.int(20, 3);
                    }
                }
                m.msg(5, am);
            }
            g.msg(1, m);
        }
        g.string(2, "prism");
        for i in &self.inits {
            let mut t = Pb::default();
            for d in &i.dims {
                t.int(1, *d);
            }
            t.int(2, i.dtype as i64);
            t.string(8, &i.name);
            t.bytes(9, &i.raw);
            g.msg(5, t);
        }
        let value_info = |name: &str, ty: i32, dims: &[Dim]| {
            let mut shape = Pb::default();
            for d in dims {
                let mut dm = Pb::default();
                match d {
                    Dim::Fixed(v) => dm.int(1, *v),
                    Dim::Named(s) => dm.string(2, s),
                }
                shape.msg(1, dm);
            }
            let mut tensor = Pb::default();
            tensor.int(1, ty as i64);
            tensor.msg(2, shape);
            let mut tp = Pb::default();
            tp.msg(1, tensor);
            let mut vi = Pb::default();
            vi.string(1, name);
            vi.msg(2, tp);
            vi
        };
        for (n, t, d) in &self.inputs {
            g.msg(11, value_info(n, *t, d));
        }
        for (n, t, d) in &self.outputs {
            g.msg(12, value_info(n, *t, d));
        }
        let mut model = Pb::default();
        model.int(1, 8); // ir_version 8 (ONNX 1.10+), supported by ONNX Runtime 1.10 and later
        model.string(2, "prism-video-enhancer");
        model.string(3, env!("CARGO_PKG_VERSION"));
        model.string(6, doc);
        model.msg(7, g);
        let mut opset_id = Pb::default();
        opset_id.string(1, "");
        opset_id.int(2, opset);
        model.msg(8, opset_id);
        model.0
    }
}

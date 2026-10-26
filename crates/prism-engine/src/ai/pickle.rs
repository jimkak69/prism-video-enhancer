//! Minimal reader for PyTorch zip checkpoints (`torch.save` since 1.6).
//!
//! Implements the pickle opcodes that state dicts use and rebuilds tensors from the zip's
//! `data/<key>` storages. Nothing is executed: globals are recorded by name and only the
//! known tensor/OrderedDict constructors are interpreted.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

/// Payloads of scalar variants are only read through `Debug` in error messages.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String, String),
    Storage { key: String, dtype: DType },
    Tensor(TensorMeta),
    Object,
    Mark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    Other,
}

impl DType {
    fn from_storage(name: &str) -> DType {
        match name {
            "FloatStorage" => DType::F32,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "DoubleStorage" => DType::F64,
            _ => DType::Other,
        }
    }
    fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
            DType::Other => 1,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorMeta {
    key: String,
    dtype: DType,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// A dense f32 tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub type StateDict = HashMap<String, Tensor>;

struct Machine<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: HashMap<u32, Val>,
}

impl<'a> Machine<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).ok_or_else(|| anyhow!("truncated pickle"))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.data.get(self.pos..self.pos + n).ok_or_else(|| anyhow!("truncated pickle"))?;
        self.pos += n;
        Ok(s)
    }
    fn u32le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.byte()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.data[start..self.pos - 1]).into_owned())
    }
    fn pop(&mut self) -> Result<Val> {
        self.stack.pop().ok_or_else(|| anyhow!("pickle stack underflow"))
    }
    fn pop_mark(&mut self) -> Result<Vec<Val>> {
        let i = self.stack.iter().rposition(|v| matches!(v, Val::Mark)).ok_or_else(|| anyhow!("pickle mark missing"))?;
        let items = self.stack.split_off(i + 1);
        self.stack.pop();
        Ok(items)
    }
    fn top(&mut self) -> Result<&mut Val> {
        self.stack.last_mut().ok_or_else(|| anyhow!("pickle stack empty"))
    }

    fn run(&mut self) -> Result<Val> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?;
                } // PROTO
                0x95 => {
                    self.take(8)?;
                } // FRAME
                b'.' => return self.pop(),
                b'}' => self.stack.push(Val::Dict(Vec::new())),
                b']' => self.stack.push(Val::List(Vec::new())),
                b')' => self.stack.push(Val::Tuple(Vec::new())),
                b'(' => self.stack.push(Val::Mark),
                b'N' => self.stack.push(Val::None),
                0x88 => self.stack.push(Val::Bool(true)),
                0x89 => self.stack.push(Val::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(Val::Int(v as i64))
                }
                b'K' => {
                    let v = self.byte()?;
                    self.stack.push(Val::Int(v as i64))
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
                    self.stack.push(Val::Int(v as i64))
                }
                0x8a => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, byte) in b.iter().enumerate().take(8) {
                        v |= (*byte as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Val::Int(v))
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Val::Float(v))
                }
                b'X' => {
                    let n = self.u32le()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Val::Str(s))
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Val::Str(s))
                }
                b'T' => {
                    let n = self.u32le()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Val::Str(s))
                }
                b'U' => {
                    let n = self.byte()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Val::Str(s))
                }
                b'B' => {
                    let n = self.u32le()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(b))
                }
                b'C' => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(b))
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Val::Global(module, name))
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Val::Str(m), Val::Str(n)) => self.stack.push(Val::Global(m, n)),
                        _ => bail!("STACK_GLOBAL expects strings"),
                    }
                }
                b'q' => {
                    let i = self.byte()? as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                b'r' => {
                    let i = self.u32le()?;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                0x94 => {
                    let i = self.memo.len() as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                b'h' => {
                    let i = self.byte()? as u32;
                    let v = self.memo.get(&i).cloned().ok_or_else(|| anyhow!("memo {i} missing"))?;
                    self.stack.push(v)
                }
                b'j' => {
                    let i = self.u32le()?;
                    let v = self.memo.get(&i).cloned().ok_or_else(|| anyhow!("memo {i} missing"))?;
                    self.stack.push(v)
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::Tuple(items))
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a]))
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b]))
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b, c]))
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::List(items))
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect();
                    self.stack.push(Val::Dict(pairs))
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Val::Dict(d) = self.top()? {
                        d.push((k, v));
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if let Val::Dict(d) = self.top()? {
                        for c in items.chunks(2) {
                            if c.len() == 2 {
                                d.push((c[0].clone(), c[1].clone()));
                            }
                        }
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    if let Val::List(l) = self.top()? {
                        l.push(v);
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Val::List(l) = self.top()? {
                        l.extend(items);
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(persistent_load(pid)?);
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(reduce(callable, args)?);
                }
                0x81 => {
                    let _args = self.pop()?;
                    let _cls = self.pop()?;
                    self.stack.push(Val::Object);
                }
                b'b' => {
                    // BUILD: object state (e.g. OrderedDict._metadata) is irrelevant here.
                    self.pop()?;
                }
                other => bail!("unsupported pickle opcode 0x{other:02x} at {}", self.pos - 1),
            }
        }
    }
}

fn persistent_load(pid: Val) -> Result<Val> {
    let Val::Tuple(items) = pid else { bail!("unexpected persistent id") };
    match items.as_slice() {
        [Val::Str(kind), Val::Global(_, storage), Val::Str(key), _location, _numel, ..] if kind == "storage" => {
            Ok(Val::Storage { key: key.clone(), dtype: DType::from_storage(storage) })
        }
        _ => bail!("unsupported persistent id {items:?}"),
    }
}

fn ints(v: &Val) -> Result<Vec<usize>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|i| match i {
                Val::Int(x) if *x >= 0 => Ok(*x as usize),
                _ => Err(anyhow!("expected non-negative integer")),
            })
            .collect(),
        _ => bail!("expected tuple of integers"),
    }
}

fn reduce(callable: Val, args: Val) -> Result<Val> {
    match &callable {
        Val::Global(m, n) if m == "torch._utils" && (n == "_rebuild_tensor_v2" || n == "_rebuild_tensor") => {
            let Val::Tuple(a) = args else { bail!("tensor args") };
            let (Some(Val::Storage { key, dtype }), Some(Val::Int(offset)), Some(shape), Some(stride)) = (a.first(), a.get(1), a.get(2), a.get(3)) else {
                bail!("unexpected _rebuild_tensor arguments");
            };
            Ok(Val::Tensor(TensorMeta { key: key.clone(), dtype: *dtype, offset: *offset as usize, shape: ints(shape)?, stride: ints(stride)? }))
        }
        Val::Global(m, n) if m == "collections" && n == "OrderedDict" => Ok(Val::Dict(Vec::new())),
        Val::Global(m, n) if m == "torch._utils" && n == "_rebuild_parameter" => match args {
            Val::Tuple(a) if !a.is_empty() => Ok(a[0].clone()),
            _ => bail!("unexpected _rebuild_parameter arguments"),
        },
        _ => Ok(Val::Object),
    }
}

fn read_tensor(zip: &mut zip::ZipArchive<std::fs::File>, prefix: &str, meta: &TensorMeta, cache: &mut HashMap<String, Vec<u8>>) -> Result<Tensor> {
    if meta.dtype == DType::Other {
        bail!("unsupported tensor dtype for storage {}", meta.key);
    }
    if !cache.contains_key(&meta.key) {
        let name = format!("{prefix}data/{}", meta.key);
        let mut f = zip.by_name(&name).with_context(|| format!("checkpoint entry {name}"))?;
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf)?;
        cache.insert(meta.key.clone(), buf);
    }
    let bytes = &cache[&meta.key];
    let numel: usize = meta.shape.iter().product();
    let es = meta.dtype.size();
    let at = |elem: usize| -> Result<f32> {
        let o = (meta.offset + elem) * es;
        let b = bytes.get(o..o + es).ok_or_else(|| anyhow!("tensor {} out of storage bounds", meta.key))?;
        Ok(match meta.dtype {
            DType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            DType::F16 => half::f16::from_le_bytes(b.try_into().unwrap()).to_f32(),
            DType::BF16 => half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32(),
            DType::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            DType::Other => unreachable!(),
        })
    };
    let mut data = Vec::with_capacity(numel);
    // Walk the logical index in row-major order and map it through the strides.
    let nd = meta.shape.len();
    let mut idx = vec![0usize; nd];
    for _ in 0..numel {
        let elem: usize = idx.iter().zip(&meta.stride).map(|(i, s)| i * s).sum();
        data.push(at(elem)?);
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < meta.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: meta.shape.clone(), data })
}

/// Loads every tensor of a checkpoint's state dict. Unwraps `params_ema`/`params`/`state_dict`
/// containers, preferring the EMA weights.
pub fn load_state_dict(path: &Path) -> Result<StateDict> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).with_context(|| format!("{} is not a PyTorch zip checkpoint (saved with torch >= 1.6)", path.display()))?;
    let pkl_name = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
        .find(|n| n.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("no data.pkl in {}", path.display()))?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let mut pkl = Vec::new();
    zip.by_name(&pkl_name)?.read_to_end(&mut pkl)?;
    let root = Machine { data: &pkl, pos: 0, stack: Vec::new(), memo: HashMap::new() }.run()?;

    let mut dict = root;
    for _ in 0..3 {
        let Val::Dict(entries) = &dict else { break };
        let inner = ["params_ema", "params", "state_dict", "model"]
            .iter()
            .find_map(|want| entries.iter().find(|(k, v)| matches!(k, Val::Str(s) if s == want) && matches!(v, Val::Dict(_))).map(|(_, v)| v.clone()));
        match inner {
            Some(v) => dict = v,
            None => break,
        }
    }
    let Val::Dict(entries) = dict else { bail!("checkpoint does not contain a state dict") };
    let mut cache = HashMap::new();
    let mut out = StateDict::new();
    for (k, v) in entries {
        if let (Val::Str(name), Val::Tensor(meta)) = (k, v) {
            out.insert(name.strip_prefix("module.").unwrap_or(&name).to_string(), read_tensor(&mut zip, &prefix, &meta, &mut cache)?);
        }
    }
    if out.is_empty() {
        bail!("no tensors found in {}", path.display());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a checkpoint shaped like `torch.save({'params': {'w': tensor}})` (protocol 2).
    fn write_checkpoint(path: &Path) {
        use std::io::Write;
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'q', 0];
        p.extend([b'X', 6, 0, 0, 0]);
        p.extend(b"params");
        p.push(b'}');
        p.extend([b'X', 1, 0, 0, 0, b'w']);
        // torch._utils._rebuild_tensor_v2((storage pid), 1, (2, 2), (1, 2), False, OrderedDict())
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.push(b'(');
        p.push(b'(');
        p.extend([b'X', 7, 0, 0, 0]);
        p.extend(b"storage");
        p.extend(b"ctorch\nFloatStorage\n");
        p.extend([b'X', 1, 0, 0, 0, b'0']);
        p.extend([b'X', 3, 0, 0, 0]);
        p.extend(b"cpu");
        p.extend([b'K', 5]);
        p.push(b't');
        p.push(b'Q');
        p.extend([b'K', 1]);
        p.extend([b'K', 2, b'K', 2, 0x86]);
        p.extend([b'K', 1, b'K', 2, 0x86]);
        p.push(0x89);
        p.extend(b"ccollections\nOrderedDict\n");
        p.push(b')');
        p.push(b'R');
        p.push(b't');
        p.push(b'R');
        p.push(b's');
        p.push(b's');
        p.push(b'.');
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        z.start_file("archive/data.pkl", opts).unwrap();
        z.write_all(&p).unwrap();
        z.start_file("archive/data/0", opts).unwrap();
        for v in [9.0f32, 1.0, 2.0, 3.0, 4.0] {
            z.write_all(&v.to_le_bytes()).unwrap();
        }
        z.finish().unwrap();
    }

    #[test]
    fn reads_strided_tensor_from_nested_params() {
        let path = std::env::temp_dir().join(format!("prism-ckpt-{}.pth", std::process::id()));
        write_checkpoint(&path);
        let sd = load_state_dict(&path).unwrap();
        let t = &sd["w"];
        assert_eq!(t.shape, vec![2, 2]);
        // offset 1, stride (1, 2): [[1, 3], [2, 4]]
        assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
        std::fs::remove_file(path).ok();
    }
}

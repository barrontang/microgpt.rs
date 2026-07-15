use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::rc::Rc;

#[derive(Clone)]
struct Value(Rc<RefCell<Node>>);

struct Node {
    data: f64,
    grad: f64,
    children: Vec<Value>,
    local_grads: Vec<f64>,
}

impl Value {
    fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        })))
    }

    fn with_graph(data: f64, children: Vec<Value>, local_grads: Vec<f64>) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    fn data(&self) -> f64 {
        self.0.borrow().data
    }

    fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn set_data(&self, x: f64) {
        self.0.borrow_mut().data = x;
    }

    fn add_grad(&self, x: f64) {
        self.0.borrow_mut().grad += x;
    }

    fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    fn ptr_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    fn add(&self, other: &Value) -> Value {
        Value::with_graph(
            self.data() + other.data(),
            vec![self.clone(), other.clone()],
            vec![1.0, 1.0],
        )
    }

    fn sub(&self, other: &Value) -> Value {
        self.add(&other.mul_scalar(-1.0))
    }

    fn mul(&self, other: &Value) -> Value {
        Value::with_graph(
            self.data() * other.data(),
            vec![self.clone(), other.clone()],
            vec![other.data(), self.data()],
        )
    }

    fn mul_scalar(&self, s: f64) -> Value {
        Value::with_graph(self.data() * s, vec![self.clone()], vec![s])
    }

    fn div_scalar(&self, s: f64) -> Value {
        self.mul_scalar(1.0 / s)
    }

    fn powf(&self, p: f64) -> Value {
        let base = self.data();
        Value::with_graph(base.powf(p), vec![self.clone()], vec![p * base.powf(p - 1.0)])
    }

    fn log(&self) -> Value {
        let x = self.data();
        Value::with_graph(x.ln(), vec![self.clone()], vec![1.0 / x])
    }

    fn exp(&self) -> Value {
        let e = self.data().exp();
        Value::with_graph(e, vec![self.clone()], vec![e])
    }

    fn relu(&self) -> Value {
        let x = self.data();
        Value::with_graph(x.max(0.0), vec![self.clone()], vec![if x > 0.0 { 1.0 } else { 0.0 }])
    }

    fn backward(&self) {
        fn build_topo(v: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
            let id = v.ptr_id();
            if visited.contains(&id) {
                return;
            }
            visited.insert(id);
            let children = v.0.borrow().children.clone();
            for ch in children {
                build_topo(&ch, visited, topo);
            }
            topo.push(v.clone());
        }

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;
        for v in topo.into_iter().rev() {
            let vg = v.grad();
            let (children, locals) = {
                let n = v.0.borrow();
                (n.children.clone(), n.local_grads.clone())
            };
            for (child, local_grad) in children.into_iter().zip(locals.into_iter()) {
                child.add_grad(local_grad * vg);
            }
        }
    }
}

type Matrix = Vec<Vec<Value>>;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller transform
        let u1 = (1.0 - self.next_f64()).max(1e-12);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z0
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_f64() * ((i + 1) as f64)) as usize;
            xs.swap(i, j);
        }
    }

    fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= *w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len().saturating_sub(1)
    }
}

fn matrix(rng: &mut Rng, nout: usize, nin: usize, std: f64) -> Matrix {
    let mut out = Vec::with_capacity(nout);
    for _ in 0..nout {
        let mut row = Vec::with_capacity(nin);
        for _ in 0..nin {
            row.push(Value::new(rng.gauss(0.0, std)));
        }
        out.push(row);
    }
    out
}

fn linear(x: &[Value], w: &Matrix) -> Vec<Value> {
    let mut out = Vec::with_capacity(w.len());
    for wo in w {
        let mut s = Value::new(0.0);
        for (wi, xi) in wo.iter().zip(x.iter()) {
            s = s.add(&wi.mul(xi));
        }
        out.push(s);
    }
    out
}

fn softmax(logits: &[Value]) -> Vec<Value> {
    let mut max_val = f64::NEG_INFINITY;
    for v in logits {
        if v.data() > max_val {
            max_val = v.data();
        }
    }

    let mut exps = Vec::with_capacity(logits.len());
    for v in logits {
        exps.push(v.sub(&Value::new(max_val)).exp());
    }

    let mut total = Value::new(0.0);
    for e in &exps {
        total = total.add(e);
    }

    let mut probs = Vec::with_capacity(exps.len());
    for e in &exps {
        probs.push(e.mul(&total.powf(-1.0)));
    }
    probs
}

fn rmsnorm(x: &[Value]) -> Vec<Value> {
    let mut ms = Value::new(0.0);
    for xi in x {
        ms = ms.add(&xi.mul(xi));
    }
    ms = ms.div_scalar(x.len() as f64);
    let scale = ms.add(&Value::new(1e-5)).powf(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    state_dict: &HashMap<String, Matrix>,
    keys: &mut [Vec<Vec<Value>>],
    values: &mut [Vec<Vec<Value>>],
) -> Vec<Value> {
    let tok_emb = &state_dict["wte"][token_id];
    let pos_emb = &state_dict["wpe"][pos_id];
    let mut x: Vec<Value> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| t.add(p))
        .collect();
    x = rmsnorm(&x);

    for li in 0..n_layer {
        let x_residual = x.clone();
        x = rmsnorm(&x);

        let q = linear(&x, &state_dict[&format!("layer{}.attn_wq", li)]);
        let k = linear(&x, &state_dict[&format!("layer{}.attn_wk", li)]);
        let v = linear(&x, &state_dict[&format!("layer{}.attn_wv", li)]);
        keys[li].push(k.clone());
        values[li].push(v.clone());

        let mut x_attn = Vec::new();
        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<Vec<Value>> = keys[li]
                .iter()
                .map(|ki| ki[hs..hs + head_dim].to_vec())
                .collect();
            let v_h: Vec<Vec<Value>> = values[li]
                .iter()
                .map(|vi| vi[hs..hs + head_dim].to_vec())
                .collect();

            let mut attn_logits = Vec::with_capacity(k_h.len());
            for kh_t in &k_h {
                let mut dot = Value::new(0.0);
                for j in 0..head_dim {
                    dot = dot.add(&q_h[j].mul(&kh_t[j]));
                }
                attn_logits.push(dot.div_scalar((head_dim as f64).sqrt()));
            }

            let attn_weights = softmax(&attn_logits);
            let mut head_out = Vec::with_capacity(head_dim);
            for j in 0..head_dim {
                let mut s = Value::new(0.0);
                for t in 0..v_h.len() {
                    s = s.add(&attn_weights[t].mul(&v_h[t][j]));
                }
                head_out.push(s);
            }
            x_attn.extend(head_out);
        }

        x = linear(&x_attn, &state_dict[&format!("layer{}.attn_wo", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc1", li)]);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc2", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();
    }

    linear(&x, &state_dict["lm_head"])
}

fn main() {
    let mut rng = Rng::new(42);

    let text = fs::read_to_string("input.txt")
        .expect("input.txt not found. Place dataset in input.txt before running.");
    let mut docs: Vec<String> = text
        .lines()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    let all_chars: String = docs.join("");
    let mut uchars: Vec<char> = all_chars.chars().collect::<HashSet<char>>().into_iter().collect();
    uchars.sort_unstable();

    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    let n_layer = 1usize;
    let n_embd = 16usize;
    let block_size = 16usize;
    let n_head = 4usize;
    let head_dim = n_embd / n_head;

    let mut state_dict: HashMap<String, Matrix> = HashMap::new();
    state_dict.insert("wte".to_string(), matrix(&mut rng, vocab_size, n_embd, 0.08));
    state_dict.insert("wpe".to_string(), matrix(&mut rng, block_size, n_embd, 0.08));
    state_dict.insert(
        "lm_head".to_string(),
        matrix(&mut rng, vocab_size, n_embd, 0.08),
    );

    for i in 0..n_layer {
        state_dict.insert(
            format!("layer{}.attn_wq", i),
            matrix(&mut rng, n_embd, n_embd, 0.08),
        );
        state_dict.insert(
            format!("layer{}.attn_wk", i),
            matrix(&mut rng, n_embd, n_embd, 0.08),
        );
        state_dict.insert(
            format!("layer{}.attn_wv", i),
            matrix(&mut rng, n_embd, n_embd, 0.08),
        );
        state_dict.insert(
            format!("layer{}.attn_wo", i),
            matrix(&mut rng, n_embd, n_embd, 0.08),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc1", i),
            matrix(&mut rng, 4 * n_embd, n_embd, 0.08),
        );
        state_dict.insert(
            format!("layer{}.mlp_fc2", i),
            matrix(&mut rng, n_embd, 4 * n_embd, 0.08),
        );
    }

    let params: Vec<Value> = state_dict
        .values()
        .flat_map(|mat| mat.iter().flat_map(|row| row.iter().cloned()))
        .collect();
    println!("num params: {}", params.len());

    let learning_rate = 0.01;
    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps_adam = 1e-8;
    let mut m = vec![0.0f64; params.len()];
    let mut v = vec![0.0f64; params.len()];

    let num_steps = 1000usize;
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];

        let mut tokens = vec![bos];
        for ch in doc.chars() {
            let idx = uchars
                .iter()
                .position(|x| *x == ch)
                .expect("character must exist in vocab");
            tokens.push(idx);
        }
        tokens.push(bos);

        let n = block_size.min(tokens.len().saturating_sub(1));
        let mut keys: Vec<Vec<Vec<Value>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Vec<Value>>> = (0..n_layer).map(|_| Vec::new()).collect();

        let mut losses = Vec::with_capacity(n);
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &state_dict,
                &mut keys,
                &mut values,
            );
            let probs = softmax(&logits);
            losses.push(probs[target_id].log().mul_scalar(-1.0));
        }

        let mut loss = Value::new(0.0);
        for lt in &losses {
            loss = loss.add(lt);
        }
        loss = loss.div_scalar(n as f64);

        loss.backward();

        let lr_t = learning_rate * (1.0 - (step as f64) / (num_steps as f64));
        for (i, p) in params.iter().enumerate() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad();
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad() * p.grad();
            let m_hat = m[i] / (1.0 - beta1.powf((step + 1) as f64));
            let v_hat = v[i] / (1.0 - beta2.powf((step + 1) as f64));
            p.set_data(p.data() - lr_t * m_hat / (v_hat.sqrt() + eps_adam));
            p.zero_grad();
        }

        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss.data());
    }

    println!("\n--- inference (new, hallucinated names) ---");
    let temperature = 0.5;
    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Vec<Value>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Vec<Value>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..block_size {
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &state_dict,
                &mut keys,
                &mut values,
            );

            let temp_logits: Vec<Value> = logits.iter().map(|l| l.div_scalar(temperature)).collect();
            let probs = softmax(&temp_logits);
            let prob_vals: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            token_id = rng.weighted_choice(&prob_vals);
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}

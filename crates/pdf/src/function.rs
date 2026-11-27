//! PDF functions (ISO 32000-2 section 7.10): `m` numbers in, `n` numbers out, read once from their object and then
//! evaluated as often as asked. A color space's tint transform is one (8.6.6.4); a shading's colors are another.
//!
//! - Type 0, sampled: a table of samples over the input domain, read through `/Encode` and `/Decode` and
//!   interpolated multilinearly between its neighbors (`/Order` 1; the optional cubic `/Order` 3 is refused).
//! - Type 2, exponential: `C0 + x^N (C1 - C0)`.
//! - Type 3, stitching: one input's domain cut by `/Bounds`, each piece mapped by `/Encode` onto a function of its
//!   own.
//! - Type 4, PostScript calculator: a program of the operators section 7.10.5 allows, parsed once into a tree (its
//!   only control flow is `if`/`ifelse`, so it runs each operator at most once). It is bounded: an evaluation
//!   refuses a stack deeper than [`STACK`] values or more than [`STEPS`] operators, and the parser a program nested
//!   deeper than [`NESTING`].
//!
//! Inputs are clipped to the domain and outputs to the range, as the standard says; anything else that is not a
//! number where one must be -- a short sample table, a program leaving the wrong count of values, a result that is
//! not finite -- is refused, never filled in.

use crate::error::{Error, Result};
use crate::file::File;
use crate::object::{Dict, Object};

/// The most values a type 4 program's operand stack holds (the limit section 7.10.5.1 lets a reader set: 100).
pub const STACK: usize = 100;
/// The most operators one evaluation of a type 4 program runs.
pub const STEPS: usize = 100_000;
/// The deepest a type 4 program's procedures nest, and a stitching function's functions.
pub const NESTING: usize = 32;

/// A function, read: its domain, its range when stated, and how it maps one to the other.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    domain: Vec<[f64; 2]>,
    range: Option<Vec<[f64; 2]>>,
    kind: Kind,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Sampled(Sampled),
    Exponential {
        c0: Vec<f64>,
        c1: Vec<f64>,
        n: f64,
    },
    Stitching {
        functions: Vec<Function>,
        bounds: Vec<f64>,
        encode: Vec<[f64; 2]>,
    },
    Calculator(Vec<Instruction>),
}

/// A type 0 function's table: its size along each input, each input's encoding, and the samples already decoded
/// (`/Decode` applied), the first input varying fastest, the outputs of a sample side by side.
#[derive(Debug, Clone, PartialEq)]
struct Sampled {
    size: Vec<usize>,
    encode: Vec<[f64; 2]>,
    outputs: usize,
    samples: Vec<f64>,
}

impl Function {
    /// How many numbers it takes.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.domain.len()
    }

    /// How many numbers it gives.
    #[must_use]
    pub fn outputs(&self) -> usize {
        match (&self.range, &self.kind) {
            (Some(range), _) => range.len(),
            (None, Kind::Sampled(s)) => s.outputs,
            (None, Kind::Exponential { c0, .. }) => c0.len(),
            (None, Kind::Stitching { functions, .. }) => functions.first().map_or(0, Self::outputs),
            (None, Kind::Calculator(_)) => 0, // a type 4 function always states its range
        }
    }

    /// Its outputs for `input`, clipped to the domain first and to the range after.
    ///
    /// # Errors
    ///
    /// Refuses an input of the wrong length, and an evaluation that yields no number (a type 4 program's error, a
    /// result that is not finite).
    pub fn evaluate(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.inputs() {
            return Err(Error::malformed(format!(
                "a function of {} inputs is given {}",
                self.inputs(),
                input.len()
            )));
        }
        let x: Vec<f64> = input
            .iter()
            .zip(&self.domain)
            .map(|(v, d)| clip(*v, *d))
            .collect();
        let mut out = match &self.kind {
            Kind::Sampled(s) => s.evaluate(&x, &self.domain),
            Kind::Exponential { c0, c1, n } => exponential(x[0], c0, c1, *n)?,
            Kind::Stitching {
                functions,
                bounds,
                encode,
            } => stitch(x[0], self.domain[0], functions, bounds, encode)?,
            Kind::Calculator(program) => calculate(program, &x, self.outputs())?,
        };
        if let Some(range) = &self.range {
            for (v, r) in out.iter_mut().zip(range) {
                *v = clip(*v, *r);
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::malformed(
                "a function's result is not a finite number",
            ));
        }
        Ok(out)
    }
}

fn clip(v: f64, [low, high]: [f64; 2]) -> f64 {
    v.max(low).min(high)
}

/// `x` in `[x0, x1]` mapped linearly onto `[y0, y1]` (7.10.2); `y0` when the first is a point.
#[expect(clippy::float_cmp, reason = "only a point, exactly, has no slope")]
fn interpolate(x: f64, [x0, x1]: [f64; 2], [y0, y1]: [f64; 2]) -> f64 {
    if x1 == x0 {
        y0
    } else {
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

impl Sampled {
    fn evaluate(&self, x: &[f64], domain: &[[f64; 2]]) -> Vec<f64> {
        // Each input's place in the table: a whole index and how far towards the next sample it lies.
        let mut base = 0;
        let mut stride = 1;
        let mut moving = Vec::new(); // (stride, share) of the inputs between two samples
        for (i, &size) in self.size.iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "a table size is far below 2^53")]
            let last = (size - 1) as f64;
            let e = interpolate(x[i], domain[i], self.encode[i]).clamp(0.0, last);
            let floor = e.floor().min((last - 1.0).max(0.0));
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "floor of a value in 0..=size-1"
            )]
            let index = floor as usize;
            base += index * stride;
            let share = e - floor;
            if share > 0.0 {
                moving.push((stride, share));
            }
            stride *= size;
        }
        let mut out = vec![0.0; self.outputs];
        // The corners of the cell the inputs fall in, weighted by how near each is (multilinear interpolation).
        for corner in 0..1_usize << moving.len() {
            let mut at = base;
            let mut weight = 1.0;
            for (bit, (stride, share)) in moving.iter().enumerate() {
                if corner >> bit & 1 == 1 {
                    at += stride;
                    weight *= share;
                } else {
                    weight *= 1.0 - share;
                }
            }
            let sample = &self.samples[at * self.outputs..(at + 1) * self.outputs];
            for (o, s) in out.iter_mut().zip(sample) {
                *o += weight * s;
            }
        }
        out
    }
}

fn exponential(x: f64, c0: &[f64], c1: &[f64], n: f64) -> Result<Vec<f64>> {
    let power = x.powf(n);
    if !power.is_finite() {
        return Err(Error::malformed(format!(
            "a type 2 function has no value at {x} to the power {n}"
        )));
    }
    Ok(c0
        .iter()
        .zip(c1)
        .map(|(a, b)| a + power * (b - a))
        .collect())
}

fn stitch(
    x: f64,
    domain: [f64; 2],
    functions: &[Function],
    bounds: &[f64],
    encode: &[[f64; 2]],
) -> Result<Vec<f64>> {
    // The piece x falls in: bounds[i-1] <= x < bounds[i], the last piece closed at the domain's end and the first at
    // its start (a first piece of one point, `Domain0 = Bounds0`, takes that point).
    let i = if x <= domain[0] {
        0
    } else {
        bounds.iter().take_while(|b| x >= **b).count()
    };
    let low = if i == 0 { domain[0] } else { bounds[i - 1] };
    let high = bounds.get(i).copied().unwrap_or(domain[1]);
    functions[i].evaluate(&[interpolate(x, [low, high], encode[i])])
}

/// A number or a boolean on a type 4 program's stack.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Integer(i64),
    Real(f64),
    Bool(bool),
}

impl Value {
    fn number(self) -> Result<f64> {
        match self {
            #[expect(clippy::cast_precision_loss, reason = "PostScript integers are 32-bit")]
            Self::Integer(i) => Ok(i as f64),
            Self::Real(r) => Ok(r),
            Self::Bool(_) => Err(calculator("a boolean where a number must be")),
        }
    }

    fn integer(self) -> Result<i64> {
        match self {
            Self::Integer(i) => Ok(i),
            _ => Err(calculator("a non-integer where an integer must be")),
        }
    }

    fn boolean(self) -> Result<bool> {
        match self {
            Self::Bool(b) => Ok(b),
            _ => Err(calculator("a number where a boolean must be")),
        }
    }
}

/// One step of a type 4 program.
#[derive(Debug, Clone, PartialEq)]
enum Instruction {
    Push(Value),
    Operator(Operator),
    If(Vec<Instruction>),
    IfElse(Vec<Instruction>, Vec<Instruction>),
}

/// The operators a type 4 program may use (7.10.5.2, Table 42), but `if` and `ifelse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Abs,
    Add,
    Atan,
    Ceiling,
    Cos,
    Cvi,
    Cvr,
    Div,
    Exp,
    Floor,
    Idiv,
    Ln,
    Log,
    Mod,
    Mul,
    Neg,
    Round,
    Sin,
    Sqrt,
    Sub,
    Truncate,
    And,
    Bitshift,
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Ne,
    Not,
    Or,
    Xor,
    Copy,
    Dup,
    Exch,
    Index,
    Pop,
    Roll,
}

impl Operator {
    fn named(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"abs" => Self::Abs,
            b"add" => Self::Add,
            b"atan" => Self::Atan,
            b"ceiling" => Self::Ceiling,
            b"cos" => Self::Cos,
            b"cvi" => Self::Cvi,
            b"cvr" => Self::Cvr,
            b"div" => Self::Div,
            b"exp" => Self::Exp,
            b"floor" => Self::Floor,
            b"idiv" => Self::Idiv,
            b"ln" => Self::Ln,
            b"log" => Self::Log,
            b"mod" => Self::Mod,
            b"mul" => Self::Mul,
            b"neg" => Self::Neg,
            b"round" => Self::Round,
            b"sin" => Self::Sin,
            b"sqrt" => Self::Sqrt,
            b"sub" => Self::Sub,
            b"truncate" => Self::Truncate,
            b"and" => Self::And,
            b"bitshift" => Self::Bitshift,
            b"eq" => Self::Eq,
            b"ge" => Self::Ge,
            b"gt" => Self::Gt,
            b"le" => Self::Le,
            b"lt" => Self::Lt,
            b"ne" => Self::Ne,
            b"not" => Self::Not,
            b"or" => Self::Or,
            b"xor" => Self::Xor,
            b"copy" => Self::Copy,
            b"dup" => Self::Dup,
            b"exch" => Self::Exch,
            b"index" => Self::Index,
            b"pop" => Self::Pop,
            b"roll" => Self::Roll,
            _ => return None,
        })
    }
}

fn calculator(message: &str) -> Error {
    Error::malformed(format!("type 4 function: {message}"))
}

/// A type 4 program's tokens: `{`, `}`, numbers, booleans and operator names; `%` comments dropped.
fn tokens(program: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < program.len() {
        let c = program[i];
        if c.is_ascii_whitespace() || c == 0 {
            i += 1;
        } else if c == b'%' {
            while i < program.len() && program[i] != b'\n' && program[i] != b'\r' {
                i += 1;
            }
        } else if c == b'{' || c == b'}' {
            out.push(&program[i..=i]);
            i += 1;
        } else {
            let start = i;
            while i < program.len()
                && !program[i].is_ascii_whitespace()
                && !matches!(program[i], b'{' | b'}' | b'%' | 0)
            {
                i += 1;
            }
            out.push(&program[start..i]);
        }
    }
    out
}

/// A type 4 program, `{ ... }`, parsed.
fn parse_program(program: &[u8]) -> Result<Vec<Instruction>> {
    let tokens = tokens(program);
    let mut at = 0;
    if tokens.first() != Some(&&b"{"[..]) {
        return Err(calculator("the program does not open with `{`"));
    }
    at += 1;
    let body = parse_block(&tokens, &mut at, 1)?;
    if at != tokens.len() {
        return Err(calculator("text follows the program's closing `}`"));
    }
    Ok(body)
}

/// The instructions up to the `}` closing the block `tokens[*at..]` is in, at nesting `depth`.
fn parse_block(tokens: &[&[u8]], at: &mut usize, depth: usize) -> Result<Vec<Instruction>> {
    if depth > NESTING {
        return Err(calculator(&format!(
            "procedures nest deeper than {NESTING}"
        )));
    }
    let mut out = Vec::new();
    let mut procedures: Vec<Vec<Instruction>> = Vec::new();
    loop {
        let Some(&token) = tokens.get(*at) else {
            return Err(calculator("a `{` is never closed"));
        };
        *at += 1;
        match token {
            b"}" => break,
            b"{" => {
                procedures.push(parse_block(tokens, at, depth + 1)?);
            }
            b"if" => match <[_; 1]>::try_from(std::mem::take(&mut procedures)) {
                Ok([then]) => out.push(Instruction::If(then)),
                Err(_) => return Err(calculator("`if` needs one procedure before it")),
            },
            b"ifelse" => match <[_; 2]>::try_from(std::mem::take(&mut procedures)) {
                Ok([then, otherwise]) => out.push(Instruction::IfElse(then, otherwise)),
                Err(_) => return Err(calculator("`ifelse` needs two procedures before it")),
            },
            _ if !procedures.is_empty() => {
                return Err(calculator(
                    "a procedure stands where only `if` or `ifelse` may follow",
                ));
            }
            b"true" => out.push(Instruction::Push(Value::Bool(true))),
            b"false" => out.push(Instruction::Push(Value::Bool(false))),
            _ => out.push(match Operator::named(token) {
                Some(op) => Instruction::Operator(op),
                None => Instruction::Push(number(token)?),
            }),
        }
    }
    if procedures.is_empty() {
        Ok(out)
    } else {
        Err(calculator("a procedure is not taken by `if` or `ifelse`"))
    }
}

/// A numeric token: an integer, or a real (a point or an exponent).
fn number(token: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(token).unwrap_or("");
    if let Ok(i) = text.parse::<i64>() {
        return Ok(Value::Integer(i));
    }
    match text.parse::<f64>() {
        Ok(r) if r.is_finite() && !text.is_empty() => Ok(Value::Real(r)),
        _ => Err(calculator(&format!(
            "`{}` is neither a number nor an operator it allows",
            String::from_utf8_lossy(token)
        ))),
    }
}

/// A program's run: its stack and how many operators it has run.
struct Machine {
    stack: Vec<Value>,
    steps: usize,
}

fn calculate(program: &[Instruction], input: &[f64], outputs: usize) -> Result<Vec<f64>> {
    let mut machine = Machine {
        stack: Vec::with_capacity(STACK),
        steps: 0,
    };
    for &x in input {
        machine.push(Value::Real(x))?;
    }
    machine.run(program)?;
    if machine.stack.len() != outputs {
        return Err(calculator(&format!(
            "the program leaves {} values for {outputs} outputs",
            machine.stack.len()
        )));
    }
    machine.stack.iter().map(|v| v.number()).collect()
}

impl Machine {
    fn push(&mut self, v: Value) -> Result<()> {
        if self.stack.len() >= STACK {
            return Err(calculator(&format!("its stack grows past {STACK} values")));
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| calculator("an operator finds too few values on the stack"))
    }

    fn real(&mut self, r: f64) -> Result<()> {
        if r.is_finite() {
            self.push(Value::Real(r))
        } else {
            Err(calculator("a result is not a finite number"))
        }
    }

    fn run(&mut self, program: &[Instruction]) -> Result<()> {
        for instruction in program {
            self.steps += 1;
            if self.steps > STEPS {
                return Err(calculator(&format!("it runs more than {STEPS} operators")));
            }
            match instruction {
                Instruction::Push(v) => self.push(*v)?,
                Instruction::Operator(op) => self.operate(*op)?,
                Instruction::If(then) => {
                    if self.pop()?.boolean()? {
                        self.run(then)?;
                    }
                }
                Instruction::IfElse(then, otherwise) => {
                    if self.pop()?.boolean()? {
                        self.run(then)?;
                    } else {
                        self.run(otherwise)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// An arithmetic operator on two numbers: integers stay integers where `whole` gives one (no overflow).
    fn arithmetic(
        &mut self,
        whole: fn(i64, i64) -> Option<i64>,
        real: fn(f64, f64) -> f64,
    ) -> Result<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        if let (Value::Integer(x), Value::Integer(y)) = (a, b)
            && let Some(v) = whole(x, y)
        {
            return self.push(Value::Integer(v));
        }
        self.real(real(a.number()?, b.number()?))
    }

    /// A one-number operator: an integer kept as one by `whole`, a real mapped by `real`.
    fn unary(&mut self, whole: fn(i64) -> Option<i64>, real: fn(f64) -> f64) -> Result<()> {
        match self.pop()? {
            Value::Integer(i) => match whole(i) {
                Some(v) => self.push(Value::Integer(v)),
                #[expect(clippy::cast_precision_loss, reason = "PostScript integers are 32-bit")]
                None => self.real(real(i as f64)),
            },
            v => self.real(real(v.number()?)),
        }
    }

    fn compare(&mut self, holds: fn(std::cmp::Ordering) -> bool) -> Result<()> {
        let b = self.pop()?.number()?;
        let a = self.pop()?.number()?;
        let order = a
            .partial_cmp(&b)
            .ok_or_else(|| calculator("numbers that do not compare"))?;
        self.push(Value::Bool(holds(order)))
    }

    fn logic(
        &mut self,
        booleans: fn(bool, bool) -> bool,
        integers: fn(i64, i64) -> i64,
    ) -> Result<()> {
        let b = self.pop()?;
        let a = self.pop()?;
        match (a, b) {
            (Value::Bool(x), Value::Bool(y)) => self.push(Value::Bool(booleans(x, y))),
            (Value::Integer(x), Value::Integer(y)) => self.push(Value::Integer(integers(x, y))),
            _ => Err(calculator(
                "a logical operator needs two booleans or two integers",
            )),
        }
    }

    fn count(&mut self) -> Result<usize> {
        usize::try_from(self.pop()?.integer()?)
            .map_err(|_| calculator("a negative count of stack values"))
    }

    fn operate(&mut self, op: Operator) -> Result<()> {
        use Operator as O;
        match op {
            O::Eq
            | O::Ne
            | O::Ge
            | O::Gt
            | O::Le
            | O::Lt
            | O::And
            | O::Or
            | O::Xor
            | O::Not
            | O::Bitshift => self.relate(op),
            O::Dup | O::Pop | O::Exch | O::Copy | O::Index | O::Roll => self.shuffle(op),
            _ => self.compute(op),
        }
    }

    /// An arithmetic operator.
    fn compute(&mut self, op: Operator) -> Result<()> {
        use Operator as O;
        match op {
            O::Add => self.arithmetic(i64::checked_add, |a, b| a + b),
            O::Sub => self.arithmetic(i64::checked_sub, |a, b| a - b),
            O::Mul => self.arithmetic(i64::checked_mul, |a, b| a * b),
            O::Div => {
                let b = self.pop()?.number()?;
                let a = self.pop()?.number()?;
                if b == 0.0 {
                    return Err(calculator("`div` by zero"));
                }
                self.real(a / b)
            }
            O::Idiv | O::Mod => {
                let b = self.pop()?.integer()?;
                let a = self.pop()?.integer()?;
                let v = if op == O::Idiv {
                    a.checked_div(b)
                } else {
                    a.checked_rem(b)
                };
                self.push(Value::Integer(
                    v.ok_or_else(|| calculator("`idiv` or `mod` by zero"))?,
                ))
            }
            O::Abs => self.unary(i64::checked_abs, f64::abs),
            O::Neg => self.unary(i64::checked_neg, |r| -r),
            O::Ceiling => self.unary(Some, f64::ceil),
            O::Floor => self.unary(Some, f64::floor),
            O::Round => self.unary(Some, |r| (r + 0.5).floor()), // PostScript rounds a half up
            O::Truncate => self.unary(Some, f64::trunc),
            O::Sqrt => self.root(),
            O::Sin => self.angle(f64::sin),
            O::Cos => self.angle(f64::cos),
            O::Atan => {
                let den = self.pop()?.number()?;
                let num = self.pop()?.number()?;
                if num == 0.0 && den == 0.0 {
                    return Err(calculator("`atan` of 0 over 0"));
                }
                self.real(num.atan2(den).to_degrees().rem_euclid(360.0))
            }
            O::Exp => {
                let exponent = self.pop()?.number()?;
                let base = self.pop()?.number()?;
                self.real(base.powf(exponent))
            }
            O::Ln | O::Log => {
                let v = self.pop()?.number()?;
                if v <= 0.0 {
                    return Err(calculator("the logarithm of a number not above 0"));
                }
                self.real(if op == O::Ln { v.ln() } else { v.log10() })
            }
            O::Cvi => {
                let v = self.pop()?.number()?.trunc();
                if !(-2_147_483_648.0..=2_147_483_647.0).contains(&v) {
                    return Err(calculator("`cvi` of a number past the integers"));
                }
                #[expect(clippy::cast_possible_truncation, reason = "checked to be in range")]
                self.push(Value::Integer(v as i64))
            }
            O::Cvr => {
                let v = self.pop()?.number()?;
                self.real(v)
            }
            _ => unreachable!("`operate` gives `compute` only arithmetic operators"),
        }
    }

    /// A relational, boolean or bitwise operator.
    fn relate(&mut self, op: Operator) -> Result<()> {
        use Operator as O;
        match op {
            O::Eq | O::Ne => {
                let b = self.pop()?;
                let a = self.pop()?;
                let same = match (a, b) {
                    (Value::Bool(x), Value::Bool(y)) => x == y,
                    (Value::Bool(_), _) | (_, Value::Bool(_)) => false,
                    #[expect(clippy::float_cmp, reason = "PostScript `eq` compares exactly")]
                    (x, y) => x.number()? == y.number()?,
                };
                self.push(Value::Bool(same == (op == O::Eq)))
            }
            O::Ge => self.compare(std::cmp::Ordering::is_ge),
            O::Gt => self.compare(std::cmp::Ordering::is_gt),
            O::Le => self.compare(std::cmp::Ordering::is_le),
            O::Lt => self.compare(std::cmp::Ordering::is_lt),
            O::And => self.logic(|a, b| a && b, |a, b| a & b),
            O::Or => self.logic(|a, b| a || b, |a, b| a | b),
            O::Xor => self.logic(|a, b| a ^ b, |a, b| a ^ b),
            O::Not => match self.pop()? {
                Value::Bool(b) => self.push(Value::Bool(!b)),
                Value::Integer(i) => self.push(Value::Integer(!i)),
                Value::Real(_) => Err(calculator("`not` of a real")),
            },
            O::Bitshift => {
                let shift = self.pop()?.integer()?;
                let v = self.pop()?.integer()?;
                let bits = u32::try_from(shift.unsigned_abs().min(63)).unwrap_or(63);
                self.push(Value::Integer(if shift >= 0 {
                    v << bits
                } else {
                    v >> bits
                }))
            }
            _ => unreachable!("`operate` gives `relate` only relational and logical operators"),
        }
    }

    /// A stack operator.
    fn shuffle(&mut self, op: Operator) -> Result<()> {
        use Operator as O;
        match op {
            O::Dup => {
                let v = *self
                    .stack
                    .last()
                    .ok_or_else(|| calculator("`dup` of an empty stack"))?;
                self.push(v)
            }
            O::Pop => self.pop().map(drop),
            O::Exch => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.push(b)?;
                self.push(a)
            }
            O::Copy => {
                let n = self.count()?;
                let start = self
                    .stack
                    .len()
                    .checked_sub(n)
                    .ok_or_else(|| calculator("`copy` of more values than the stack holds"))?;
                for i in start..start + n {
                    self.push(self.stack[i])?;
                }
                Ok(())
            }
            O::Index => {
                let n = self.count()?;
                let v = self
                    .stack
                    .len()
                    .checked_sub(n + 1)
                    .map(|i| self.stack[i])
                    .ok_or_else(|| calculator("`index` past the bottom of the stack"))?;
                self.push(v)
            }
            O::Roll => {
                let j = self.pop()?.integer()?;
                let n = self.count()?;
                let start = self
                    .stack
                    .len()
                    .checked_sub(n)
                    .ok_or_else(|| calculator("`roll` of more values than the stack holds"))?;
                if n > 0 {
                    // Rolling up by j moves the top j values below the rest.
                    let by = usize::try_from(j.rem_euclid(i64::try_from(n).unwrap_or(i64::MAX)))
                        .unwrap_or(0);
                    self.stack[start..].rotate_right(by);
                }
                Ok(())
            }
            _ => unreachable!("`operate` gives `shuffle` only stack operators"),
        }
    }

    fn root(&mut self) -> Result<()> {
        let v = self.pop()?.number()?;
        if v < 0.0 {
            return Err(calculator("`sqrt` of a negative number"));
        }
        self.real(v.sqrt())
    }

    /// A trigonometric operator, its angle in degrees.
    fn angle(&mut self, f: fn(f64) -> f64) -> Result<()> {
        let v = self.pop()?.number()?;
        self.real(f(v.to_radians()))
    }
}

/// `n` numbers of an array, or refused naming `what` and `key`.
fn numbers(value: Option<Object>, what: &str, key: &str) -> Result<Vec<f64>> {
    let refused = || Error::malformed(format!("{what}: /{key} is not an array of numbers"));
    match value {
        Some(Object::Array(items)) => items
            .iter()
            .map(|v| v.as_number().ok_or_else(refused))
            .collect(),
        _ => Err(refused()),
    }
}

/// An array of numbers taken as pairs (`/Domain`, `/Range`, `/Encode`, `/Decode`).
fn pairs(value: Option<Object>, what: &str, key: &str) -> Result<Vec<[f64; 2]>> {
    let flat = numbers(value, what, key)?;
    if flat.is_empty() || flat.len() % 2 != 0 {
        return Err(Error::malformed(format!(
            "{what}: /{key} is not pairs of numbers"
        )));
    }
    Ok(flat.as_chunks::<2>().0.to_vec())
}

impl File {
    /// The function `value` is or refers to.
    ///
    /// # Errors
    ///
    /// Refuses a function that is not one of the four types or misses what its type needs; a type 0 table
    /// shorter than its `/Size` says, or of a cubic `/Order` (unsupported); a type 4 program that does not parse.
    pub fn function(&self, value: &Object) -> Result<Function> {
        self.function_at_depth(value, 0)
    }

    fn function_at_depth(&self, value: &Object, depth: usize) -> Result<Function> {
        if depth > NESTING {
            return Err(Error::malformed(format!(
                "functions nest deeper than {NESTING}"
            )));
        }
        let value = self.resolve(value)?;
        let dict = value
            .as_dict()
            .ok_or_else(|| Error::malformed(format!("a function is a {}", value.kind())))?;
        let kind = self
            .get(dict, b"FunctionType")?
            .and_then(|t| t.as_integer());
        let what = format!("a type {} function", kind.unwrap_or(-1));
        let domain = pairs(self.get(dict, b"Domain")?, &what, "Domain")?;
        let range = match self.get(dict, b"Range")? {
            None => None,
            some => Some(pairs(some, &what, "Range")?),
        };
        let one_input = || {
            if domain.len() == 1 {
                Ok(())
            } else {
                Err(Error::malformed(format!(
                    "{what} takes one input, its /Domain says {}",
                    domain.len()
                )))
            }
        };
        let kind = match kind {
            Some(0) => Kind::Sampled(self.sampled(&value, dict, &domain, range.as_deref())?),
            Some(2) => {
                one_input()?;
                let c = |key: &[u8], name: &str, default: f64| -> Result<Vec<f64>> {
                    match self.get(dict, key)? {
                        None => Ok(vec![default]),
                        some => numbers(some, &what, name),
                    }
                };
                let (c0, c1) = (c(b"C0", "C0", 0.0)?, c(b"C1", "C1", 1.0)?);
                if c0.len() != c1.len() {
                    return Err(Error::malformed(format!(
                        "{what}: /C0 and /C1 differ in length"
                    )));
                }
                let n = self
                    .get(dict, b"N")?
                    .and_then(|n| n.as_number())
                    .ok_or_else(|| Error::malformed(format!("{what} has no /N")))?;
                Kind::Exponential { c0, c1, n }
            }
            Some(3) => {
                one_input()?;
                let Some(Object::Array(items)) = self.get(dict, b"Functions")? else {
                    return Err(Error::malformed(format!("{what} has no /Functions")));
                };
                let functions = items
                    .iter()
                    .map(|f| self.function_at_depth(f, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                let bounds = numbers(self.get(dict, b"Bounds")?, &what, "Bounds")?;
                let encode = pairs(self.get(dict, b"Encode")?, &what, "Encode")?;
                let outputs = functions.first().map(Function::outputs);
                if functions.is_empty()
                    || bounds.len() + 1 != functions.len()
                    || encode.len() != functions.len()
                    || functions
                        .iter()
                        .any(|f| f.inputs() != 1 || Some(f.outputs()) != outputs)
                {
                    return Err(Error::malformed(format!(
                        "{what}: /Functions, /Bounds and /Encode do not agree"
                    )));
                }
                Kind::Stitching {
                    functions,
                    bounds,
                    encode,
                }
            }
            Some(4) => {
                let Object::Stream(stream) = &value else {
                    return Err(Error::malformed(format!("{what} is not a stream")));
                };
                if range.is_none() {
                    return Err(Error::malformed(format!("{what} has no /Range")));
                }
                Kind::Calculator(parse_program(&self.decode(stream)?.into_bytes()?)?)
            }
            _ => return Err(Error::malformed(format!("{what} is not one PDF defines"))),
        };
        Ok(Function {
            domain,
            range,
            kind,
        })
    }

    fn sampled(
        &self,
        value: &Object,
        dict: &Dict,
        domain: &[[f64; 2]],
        range: Option<&[[f64; 2]]>,
    ) -> Result<Sampled> {
        let what = "a type 0 function";
        let Object::Stream(stream) = value else {
            return Err(Error::malformed(format!("{what} is not a stream")));
        };
        let range = range.ok_or_else(|| Error::malformed(format!("{what} has no /Range")))?;
        let size = numbers(self.get(dict, b"Size")?, what, "Size")?
            .into_iter()
            .map(|s| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "checked whole and positive first"
                )]
                (s >= 1.0 && s.fract() == 0.0 && s <= f64::from(u32::MAX)).then_some(s as usize)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::malformed(format!("{what}: /Size is not whole numbers from 1"))
            })?;
        if size.len() != domain.len() {
            return Err(Error::malformed(format!(
                "{what}: /Size and /Domain differ in length"
            )));
        }
        let bits = self
            .get(dict, b"BitsPerSample")?
            .and_then(|b| b.as_integer());
        let bits = match bits {
            Some(b @ (1 | 2 | 4 | 8 | 12 | 16 | 24 | 32)) => u32::try_from(b).unwrap_or(8),
            _ => {
                return Err(Error::malformed(format!(
                    "{what} has no valid /BitsPerSample"
                )));
            }
        };
        match self.get(dict, b"Order")?.and_then(|o| o.as_integer()) {
            None | Some(1) => {}
            Some(3) => {
                return Err(Error::unsupported(format!(
                    "{what} of cubic /Order 3 (this reader interpolates linearly only)"
                )));
            }
            Some(o) => return Err(Error::malformed(format!("{what} of /Order {o}"))),
        }
        #[expect(clippy::cast_precision_loss, reason = "a table size is far below 2^53")]
        let encode = match self.get(dict, b"Encode")? {
            None => size.iter().map(|&s| [0.0, (s - 1) as f64]).collect(),
            some => pairs(some, what, "Encode")?,
        };
        let decode = match self.get(dict, b"Decode")? {
            None => range.to_vec(),
            some => pairs(some, what, "Decode")?,
        };
        if encode.len() != size.len() || decode.len() != range.len() {
            return Err(Error::malformed(format!(
                "{what}: /Encode or /Decode do not match /Size and /Range"
            )));
        }
        let outputs = range.len();
        let count = size
            .iter()
            .try_fold(outputs, |n, &s| n.checked_mul(s))
            .filter(|n| n.checked_mul(bits as usize).is_some())
            .ok_or_else(|| Error::malformed(format!("{what}: /Size is past any table")))?;
        let data = self.decode(stream)?.into_bytes()?;
        if data.len() * 8 < count * bits as usize {
            return Err(Error::malformed(format!(
                "{what}: its table holds {} bytes, /Size and /BitsPerSample need {}",
                data.len(),
                (count * bits as usize).div_ceil(8)
            )));
        }
        let top = f64::from(u32::MAX >> (32 - bits));
        let samples = (0..count)
            .map(|i| {
                let d = decode[i % outputs];
                interpolate(f64::from(sample(&data, i, bits)), [0.0, top], d)
            })
            .collect();
        Ok(Sampled {
            size,
            encode,
            outputs,
            samples,
        })
    }
}

/// The `i`-th `bits`-bit sample of `data`, most significant bit first.
fn sample(data: &[u8], i: usize, bits: u32) -> u32 {
    let start = i * bits as usize;
    (start..start + bits as usize).fold(0, |v, bit| {
        v << 1 | u32::from(data[bit / 8] >> (7 - bit % 8) & 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(text: &str, inputs: &[f64], outputs: usize) -> Result<Vec<f64>> {
        calculate(&parse_program(text.as_bytes())?, inputs, outputs)
    }

    fn near(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
    }

    #[test]
    fn a_program_branches_with_if_and_ifelse() {
        let p = "{ dup 0.5 gt { 1 exch sub } if dup 0.25 lt { 2 mul } { 0 add } ifelse }";
        assert!(near(&program(p, &[0.9], 1).unwrap(), &[0.2]));
        assert!(near(&program(p, &[0.7], 1).unwrap(), &[0.3]));
        assert!(near(&program(p, &[0.1], 1).unwrap(), &[0.2]));
    }

    #[test]
    fn a_program_computes_with_the_operators_it_allows() {
        let cases: &[(&str, &[f64], &[f64])] = &[
            ("{ 3 2 roll }", &[1.0, 2.0, 3.0], &[2.0, 3.0, 1.0]),
            ("{ 2 copy add }", &[1.0, 2.0], &[1.0, 2.0, 3.0]),
            ("{ 1 index }", &[1.0, 2.0], &[1.0, 2.0, 1.0]),
            ("{ 1 0 atan 90 eq { 1 } { 0 } ifelse }", &[0.0], &[0.0, 1.0]),
            (
                "{ pop 7 2 idiv 7 2 mod -2.5 round 2 3 exp }",
                &[0.0],
                &[3.0, 1.0, -2.0, 8.0],
            ),
            (
                "{ pop 1 3 bitshift 8 -2 bitshift 5 3 and true false or { 1 } { 0 } ifelse }",
                &[0.0],
                &[8.0, 2.0, 1.0, 1.0],
            ),
            (
                "{ cvi 100 ln 100 log 2 sqrt 90 sin -1.5 abs 1.5 ceiling }",
                &[3.7],
                &[3.0, 100_f64.ln(), 2.0, 2_f64.sqrt(), 1.0, 1.5, 2.0],
            ),
        ];
        for (text, input, expected) in cases {
            let out = program(text, input, expected.len()).unwrap();
            assert!(near(&out, expected), "{text}: {out:?}");
        }
    }

    #[test]
    fn a_program_is_bounded() {
        let deep = format!("{{ {} }}", "dup ".repeat(STACK));
        let refused = program(&deep, &[0.0], 1).unwrap_err();
        assert!(
            refused.message.contains("stack grows past 100"),
            "{refused}"
        );
        let long = format!("{{ {} }}", "dup pop ".repeat(STEPS / 2 + 1));
        let refused = program(&long, &[0.0], 1).unwrap_err();
        assert!(
            refused.message.contains("more than 100000 operators"),
            "{refused}"
        );
        let nested = format!(
            "{{ {} true {} if }}",
            "true { ".repeat(NESTING),
            "} if ".repeat(NESTING)
        );
        assert!(
            parse_program(nested.as_bytes())
                .unwrap_err()
                .message
                .contains("nest deeper")
        );
    }

    #[test]
    fn a_program_refuses_what_it_cannot_compute() {
        for (text, cause) in [
            ("{ 0 div }", "by zero"),
            ("{ pop pop }", "too few values"),
            ("{ dup }", "leaves 2 values"),
            ("{ foo }", "neither a number nor an operator"),
            ("{ { 1 } }", "not taken by"),
            ("{ 1 { 2 } 3 if }", "only `if` or `ifelse`"),
            ("{ neg sqrt }", "negative"),
            ("{ 1 ", "never closed"),
        ] {
            let refused = program(text, &[1.0], 1).unwrap_err();
            assert!(refused.message.contains(cause), "{text}: {refused}");
        }
    }

    #[test]
    fn samples_are_read_at_every_width() {
        assert_eq!(sample(&[0b1011_0010], 1, 2), 0b11);
        assert_eq!(sample(&[0xAB, 0xCD], 1, 4), 0xB);
        assert_eq!(sample(&[0xAB, 0xCD, 0xEF], 1, 12), 0xDEF);
        assert_eq!(sample(&[0, 1, 2, 3, 4, 5], 1, 24), 0x03_0405);
        assert_eq!(sample(&[0, 0, 0, 0, 1, 2, 3, 4], 1, 32), 0x0102_0304);
    }
}

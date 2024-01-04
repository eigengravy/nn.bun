enum Op {
  ADD,
  MUL,
  TANH,
  EXP,
  POW,
}

function topological_sort(v: Value, t: Value[] = []) {
  const visited = new Set<Value>();
  if (!visited.has(v)) {
    for (const p of v._prev) {
      topological_sort(p, t);
    }
    t.push(v);
  }
  return t;
}

class Value {
  _data: number;
  _prev: Set<Value>;
  _op?: Op;
  _grad: number = 0;
  _backward: () => void = () => {};
  constructor(data: number, children?: Value[], op?: Op) {
    this._data = data;
    this._prev = new Set(children);
    this._op = op;
  }

  add(other: Value | number): Value {
    const _other = other instanceof Value ? other : new Value(other);
    const out = new Value(this._data + _other._data, [this, _other], Op.ADD);
    out._backward = () => {
      this._grad += 1.0 * out._grad;
      _other._grad += 1.0 * out._grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const _other = other instanceof Value ? other : new Value(other);
    const out = new Value(this._data * _other._data, [this, _other], Op.MUL);
    out._backward = () => {
      this._grad += _other._data * out._grad;
      _other._grad += this._data * out._grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value): Value {
    return this.add(other.neg());
  }

  tanh(): Value {
    const out = new Value(Math.tanh(this._data), [this], Op.TANH);
    out._backward = () => {
      this._grad += (1 - Math.pow(out._data, 2)) * out._grad;
    };
    return out;
  }

  exp(): Value {
    const out = new Value(Math.exp(this._data), [this], Op.EXP);
    out._backward = () => {
      this._grad += out._data * out._grad;
    };
    return out;
  }

  pow(x: number): Value {
    const out = new Value(Math.pow(this._data, x), [this], Op.POW);
    out._backward = () => {
      this._grad += x * Math.pow(out._data, x - 1) * out._grad;
    };
    return out;
  }

  print() {
    console.log(
      [
        "Value {",
        ` Data = ${this._data}`,
        ` Grad = ${this._grad}`,
        this._op != undefined ? ` Op   = ${Op[this._op!]}` : "",
        this._prev.size > 0
          ? ` Prev = ${Array.from(this._prev)
              .map((v: Value) => v._data)
              .join(" ,")}`
          : "",
        "}",
      ]
        .filter((s) => s != "")
        .join("\n")
    );
  }

  backward() {
    this._grad = 1;
    for (const value of topological_sort(this).reverse()) {
      value._backward();
    }
  }
}

class Neuron {
  w: Value[];
  b: Value;
  constructor(nin: number) {
    this.w = [...new Array(nin)].map((_) => new Value(Math.random() * 2 - 1));
    this.b = new Value(Math.random() * 2 - 1);
  }

  call(x: Value[]) {
    return this.w
      .map((wi, index) => wi.mul(x[index]))
      .reduce((a, b) => a.add(b), this.b)
      .tanh();
  }

  public get parameters(): Value[] {
    return this.w.concat([this.b]);
  }
}

class Layer {
  neurons: Neuron[];
  constructor(nin: number, nout: number) {
    this.neurons = [...new Array(nout)].map((_) => new Neuron(nin));
  }

  call(x: Value[]) {
    return this.neurons.map((n) => n.call(x));
  }

  public get parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters);
  }
}

class MLP {
  layers: Layer[];
  constructor(size: number[]) {
    this.layers = size.slice(1).map((sz, i) => new Layer(size[i], sz));
  }

  call(x: Value[]) {
    for (const layer of this.layers) {
      x = layer.call(x);
    }
    return x;
  }

  public get parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters);
  }
}

export { Value, MLP };

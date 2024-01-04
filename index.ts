import { MLP, Value } from "./nn.bun/micrograd";

const mlp = new MLP([3, 4, 4, 1]);

const xs = [
  [new Value(2), new Value(3), new Value(-1)],
  [new Value(3), new Value(-1), new Value(0.5)],
  [new Value(0.5), new Value(0.5), new Value(1)],
  [new Value(1), new Value(1), new Value(-1)],
];

const ys = [new Value(1), new Value(-1), new Value(-1), new Value(1)];

// training loop
for (let index = 0; index < 20; index++) {
  // forward
  let yx = xs.flatMap((x) => mlp.call(x));
  let loss = ys.map((y, i) => y.sub(yx[i]).pow(2)).reduce((a, b) => a.add(b));

  // zero grad
  mlp.parameters.forEach((v) => (v._grad = 0));

  // backward
  loss.backward();

  // update
  mlp.parameters.forEach((v) => (v._data += -0.001 * v._grad));

  console.log(index, loss._data);
}

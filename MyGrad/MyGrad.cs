using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Json;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace MyGrad
{
    public class Value {
        public double data = 0.0;
        public double grad = 0.0;
        public delegate void BackwardFunc();
        public BackwardFunc backward;
        public Value[] prev = new Value[0];
        public char op;
        public string label;

        public Value(double data, Value[] children = null, char op = ' ', string label = "") { 
            this.data = data;
            this.label = label;
            if (children != null) {
                prev = children;
            }
            this.op = op;
        }

        public static Value operator +(Value a, Value b) {
            Value output = new Value(a.data + b.data, new Value[] { a, b }, '+' );
            output.backward = () => {
                a.grad += 1.0 * output.grad;
                b.grad += 1.0 * output.grad;
            };

            output.label = a.label + b.label;

            return output;
        }

        public static Value operator +(Value a, double bValue) {
            Value b = new Value(bValue);
            return a + b;
        }

        public static Value operator -(Value a, Value b) {
            return a + (b * -1.0f);
        }

        public static Value operator *(Value a, Value b) {
            Value output = new Value(a.data * b.data, new Value[] { a, b }, '*');
            output.backward = () => {
                a.grad += b.data * output.grad;
                b.grad += a.data * output.grad;
            };

            output.label = a.label + b.label;

            return output;
        }

        public static Value operator *(Value a, double bValue) {
            Value b = new Value(bValue);
            return a * b;
        }

        public Value Tanh() {
            double x = data;
            double y = Math.Tanh(x);
            Value output = new Value(y, new Value[] { this }, 't');
            output.backward = () => {
                this.grad += (1.0 - y * y) * output.grad;
            };

            return output;
        }

        public Value Pow(int x) {
            Value output = new Value(Math.Pow(data, x), new Value[] { this }, '^');
            output.backward = () => {
                grad += x * Math.Pow(data, x - 1) * output.grad;
            };

            return output;
        }

        public static void PrintTopo(List<Value> topo) {
            Console.WriteLine( $"Topo(Count = {topo.Count}) = {{ ");
            foreach (Value v in topo) {
                Console.Write("\t");
                Console.WriteLine(v);
            }
            Console.WriteLine("}");
        }

        public static void BuildTopo(List<Value> topo, HashSet<Value> visited, Value v) {
            if (!visited.Contains(v)) {
                visited.Add(v);
                foreach (Value c in v.prev) {
                    BuildTopo(topo, visited, c);
                }
                topo.Add(v);
            }
        }

        public void Backward() {
            List<Value> topo = new List<Value>();
            BuildTopo(topo, new HashSet<Value>(), this);
            topo.Reverse();

            //Console.WriteLine("============BEFORE==========");
            //PrintTopo(topo);

            grad = 1.0;
            foreach (Value v in topo) {
                if (v.backward != null) {
                    v.backward();
                }
            }

            //Console.WriteLine("============AFTER-==========");
            //PrintTopo(topo);
        }

        public override string ToString() {
            string sData = string.Format("{0:0.00000}", data);
            string sGrad = string.Format("{0:0.00000}", grad);
            return $"{label} = \t\tdata={sData}, \t\tgrad={sGrad}, \t\top={op}";
        }
    }

    class Neuron {
        int n;
        public List<Value> w;
        public Value b;

        public Neuron(int n) {
            this.n = n;
            Random random = new Random();
            w = new List<Value>(n);
            for (int i = 0; i < n; i++) {
                double ww = random.NextDouble() * 2 - 1;
                w.Add(new Value(ww));
            }

            b = new Value(random.NextDouble() * 2 - 1);
        }

        public Value Call(List<Value> x) {
            // w * x + b
            Value sum = new Value(0);
            sum = sum + b;
            for (int i = 0; i < n; i++) {
                Value ww = w[i];
                Value xx = x[i];
                Value r = ww * xx;
                sum = sum + r;
            }

            return sum.Tanh();
        }
        public List<Value> Parameters() {
            List<Value> l = new List<Value>();

            l.AddRange(w);
            l.Add(b);

            return l;
        }
    }
    class Layer {
        int nin;
        int nout;
        List<Neuron> neurons;

        public Layer(int nin, int nout) {
            this.nin = nin;
            this.nout = nout;
            this.neurons = new List<Neuron>();
            for (int i = 0; i < nout; i++) {
                neurons.Add(new Neuron(nin));
            }
        }

        public List<Value> Call(List<Value> x) {
            Debug.Assert(x.Count == nin);

            List<Value> results = new List<Value>();
            foreach (Neuron n in neurons) {
                results.Add( n.Call(x) );
            }

            return results;
        }

        public List<Value> Parameters() {
            List<Value> l = new List<Value>();
            neurons.ForEach(x => l.AddRange(x.Parameters()));
            return l;
        }
    }

    class MLP {
        List<Layer> layers;
        public MLP(int nin, int[] nouts) {
            int[] ns = new int[nouts.Length + 1];
            ns[0] = nin;
            for (int i = 1; i < ns.Length; i++) {
                ns[i] = nouts[i - 1];
            }

            layers = new List<Layer>(nouts.Length);
            for (int i = 0; i < nouts.Length; i++) {
                layers.Add(new Layer(ns[i], ns[i + 1]));
            }
        }

        public List<Value> Call(List<Value> values) {
            foreach (Layer layer in layers) {
                values = layer.Call(values);
            }

            return values;
        }

        public List<Value> Parameters() {
            List<Value> l = new List<Value>();
            layers.ForEach(x => l.AddRange(x.Parameters()));
            return l;
        }
    }

    public class MyGrad
    {
        public static void Print(List<Value> values) {
            values.ForEach(x => Console.WriteLine(x));
        }

        public static void Main(string[] args) {

            //Value x1 = new Value(2.0, label:nameof(x1));
            //Value x2 = new Value(0.0, label:nameof(x2));
            //Value w1 = new Value(-3.0, label:nameof(w1));
            //Value w2 = new Value(1.0, label:nameof(w2));
            //Value x1w1 = x1 * w1;
            //Value x2w2 = x2 * w2;
            //Value sum = x1w1 + x2w2;
            //Value b = new Value(6.8813735870195432, label:nameof(b));
            //Value n = sum + b; n.label = nameof(n);
            //Value o = n.Tanh(); o.label = nameof(o);
            ////o.Backward();
            //Value a = new Value(2, label: "a");
            //Value r = a * 2;
            //Console.WriteLine(r);

            List<Value> values = new List<Value> { new Value(1), new Value(-3), new Value(10) };

            MLP mlp = new MLP(3, new int[] { 4, 4, 1 });

            int count = ((3 + 1) * 4) + ((4 + 1) * 4) + ((4 + 1) * 1);

            //Print(mlp.Parameters());
            //Console.WriteLine($"c={ c }, v = { mlp.Parameters().Count }");

            List<Value> ys = new List<Value> { new Value(0.0), new Value(-1.0), new Value(-1.0), new Value(1.0) };
            List<List<Value>> xs = new List<List<Value> > {
                new List<Value> { new Value(2.0), new Value(3.0), new Value(-1.0) },
                new List<Value>{ new Value(3.0), new Value(-1.0), new Value(0.5) },
                new List<Value>{ new Value(0.5), new Value(1.0), new Value(1.0) },
                new List<Value>{ new Value(1.0), new Value(1.0), new Value(-1.0) },
            };

            int trainCount = 20000;
            for (int i = 0; i < trainCount; i++) {
                List<Value> ypred = new List<Value>();
                foreach (List<Value> x in xs) {
                    ypred.Add(mlp.Call(x)[0]);
                }

                Value loss = new Value(0);
                for (int j = 0; j < ys.Count; j++) {
                    loss = loss + (ys[j] - ypred[j]).Pow(2);
                }

                List<Value> pm = mlp.Parameters();

                pm.ForEach(x => x.grad = 0);
                loss.Backward();

                foreach (Value p in pm) {
                    p.data -= 0.01 * p.grad;
                }

                double percent = (double)(i + 1) / trainCount;
                double maxProgressSigns = 25;
                int tokenCount = (int)(percent * maxProgressSigns);
                string progressLine = "Progress ||";
                for (int t = 0; t < tokenCount; t++) {
                    progressLine += '=';
                }
                for (int t = tokenCount; t < maxProgressSigns; t++) {
                    progressLine += '-';
                }
                progressLine += "||\r";

                Console.Write(progressLine);
                    
                if (i == trainCount - 1) {
                    Console.WriteLine();
                    Console.WriteLine("Loss = \t" + loss.data);
                    for (int j = 0; j < ypred.Count; j++) {
                        Console.WriteLine("Y" + j + " = \t" + ypred[j].data);
                    }
                }
            }

            Console.ReadLine();
        }

    }
}

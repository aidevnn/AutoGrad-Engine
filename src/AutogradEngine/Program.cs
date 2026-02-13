// =============================================================================
// MicroGPT — A complete GPT language model in pure C#, no dependencies.
//
// Faithful port of Andrej Karpathy's microgpt.py art project:
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
//
// This is the exact same algorithm behind ChatGPT, stripped to its essence.
// Real GPTs have billions of parameters and run on GPU clusters.
// This one has ~3,600 parameters and runs on a single CPU thread.
// But every conceptual piece is here. Everything else is "just" optimization.
//
// What does it do? It learns to generate fake human names by reading real ones.
// You feed it "Emma", "Oliver", "Sophia" — after training, it invents new names
// that sound real but never existed.
//
// How? The model is a pile of numbers (parameters). At the start, they're random.
// During training, you show it real names and ask: "given the letters so far,
// what comes next?" It guesses, gets it wrong, and you nudge the numbers slightly
// in a direction that would have made the guess better. Repeat 1000 times.
// =============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;

using static NeuralOps;

// =============================================================================
// PART 2: THE GPT MODEL
// =============================================================================

public class MicroGPT
{
    // -------------------------------------------------------------------------
    // HYPERPARAMETERS — the knobs that control the model's size and training.
    //
    // nEmbd:        How many numbers represent each token. More = richer
    //               representation, but more parameters to learn.
    // nLayer:       How many transformer layers stacked on top of each other.
    //               More layers = deeper reasoning, but slower.
    // blockSize:    Maximum sequence length. The model can only "see" this many
    //               tokens at once. GPT-4 uses ~128K. We use 8.
    // nHead:        Number of attention heads. Each head can focus on a different
    //               pattern (e.g., one head might track vowel patterns, another
    //               might track name length). Must divide nEmbd evenly.
    // numSteps:     How many training examples to learn from.
    // learningRate: How aggressively to adjust parameters. Too high = unstable.
    //               Too low = learns too slowly.
    // -------------------------------------------------------------------------
    static int nEmbd;
    static int nLayer;
    static int blockSize;
    static int numSteps;
    static int nHead;
    static double learningRate;
    static int seed;
    static int headDim; // = nEmbd / nHead. Each attention head works on this slice.

    static Tokenizer tokenizer = null!;

    // -------------------------------------------------------------------------
    // MODEL WEIGHTS — the actual learnable numbers.
    //
    // stateDict holds all the weight matrices, keyed by name.
    // parameters is a flat list of every individual Value — used by the optimizer
    // to update all weights in one loop.
    //
    // At the start, all weights are random noise. After training, they encode
    // the statistical patterns of English names.
    // -------------------------------------------------------------------------
    static Dictionary<string, Value[][]> stateDict = new();
    static List<Value> parameters = new();

    static Random rng = null!;

    // Simple CLI argument parser (matches Python's argparse behavior).
    static int ParseArg(string[] args, string name, int defaultVal)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == $"--{name}") return int.Parse(args[i + 1]);
        return defaultVal;
    }

    static double ParseArg(string[] args, string name, double defaultVal)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == $"--{name}") return double.Parse(args[i + 1]);
        return defaultVal;
    }

    // Generates a random number from a normal (Gaussian) distribution.
    // .NET doesn't have this built-in, so we use the Box-Muller transform:
    // take two uniform random numbers and convert them to a bell curve.
    // Neural network weights are typically initialized from a Gaussian
    // with a small standard deviation (0.02) — large initial values cause
    // training instability.
    static double Gauss(double mean, double std)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + std * normal;
    }

    // Creates a matrix (2D array) of random Value objects.
    // Each Value starts with a small random number and a gradient of 0.
    // These matrices ARE the model — every learned pattern lives in them.
    static Value[][] Matrix(int nout, int nin, double std = 0.02)
    {
        var mat = new Value[nout][];
        for (int i = 0; i < nout; i++)
        {
            mat[i] = new Value[nin];
            for (int j = 0; j < nin; j++)
                mat[i][j] = new Value(Gauss(0, std));
        }
        return mat;
    }

    // =========================================================================
    // THE GPT FUNCTION — processes one token at a time.
    // =========================================================================
    //
    // This is the brain of the model. Given a token and its position, it outputs
    // a score for every possible next token. High score = model thinks that
    // token is likely to come next.
    //
    // It processes tokens one at a time (not all at once), storing past keys
    // and values in a "KV cache" so it can attend to all previous tokens.
    //
    // Architecture for each layer:
    //   1. Normalize → Attention → Add back original (residual connection)
    //   2. Normalize → MLP → Add back original (residual connection)
    //
    // The residual connections (adding back the original) are critical.
    // They let gradients flow straight through during backprop, which prevents
    // the "vanishing gradient" problem that killed deep networks before ResNets.
    // =========================================================================
    static Value[] GPT(int tokenId, int posId, List<Value[]>[] keys, List<Value[]>[] values)
    {
        // Step 1: Look up embeddings.
        // Each token has a learned vector (its "meaning" in number-space).
        // Each position also has a learned vector (encodes where in the sequence we are).
        // Add them together: now we have a single vector that represents
        // "this specific token at this specific position."
        var tokEmb = stateDict["wte"][tokenId];
        var posEmb = stateDict["wpe"][posId % blockSize];
        var x = tokEmb.Zip(posEmb, (t, p) => t + p).ToArray();

        for (int li = 0; li < nLayer; li++)
        {
            // =================================================================
            // MULTI-HEAD SELF-ATTENTION
            // =================================================================
            //
            // The key idea: for each token, look at all previous tokens and ask
            // "which ones are relevant to predicting what comes next?"
            //
            // Three projections per token:
            //   Q (Query):  "What am I looking for?"
            //   K (Key):    "What do I contain?"
            //   V (Value):  "What information do I offer if selected?"
            //
            // Attention score = dot product of Q and K. High dot product means
            // the query and key are "compatible" — that past token is relevant.
            //
            // The output is a weighted average of V vectors, where the weights
            // are the attention scores (after softmax).
            //
            // "Multi-head" means we split the vector into chunks and run
            // independent attention on each chunk. Each head can learn to focus
            // on different patterns (e.g., one might track consonant patterns,
            // another might track name length).
            //
            // Causality is free here because the KV cache only contains tokens
            // from the past — the model can't cheat by looking at future tokens.
            // =================================================================

            var xResidual = x; // save for residual connection
            x = RMSNorm(x);

            // Project input into Q, K, V vectors
            var q = Linear(x, stateDict[$"layer{li}.attn_wq"]);
            var k = Linear(x, stateDict[$"layer{li}.attn_wk"]);
            var val = Linear(x, stateDict[$"layer{li}.attn_wv"]);

            // Store K and V in the cache so future tokens can attend to this one
            keys[li].Add(k);
            values[li].Add(val);

            var xAttn = new List<Value>();
            for (int h = 0; h < nHead; h++)
            {
                // Each head operates on its own slice of the vector
                int hs = h * headDim;
                var qH = q[hs..(hs + headDim)];
                var kH = keys[li].Select(ki => ki[hs..(hs + headDim)]).ToList();
                var vH = values[li].Select(vi => vi[hs..(hs + headDim)]).ToList();

                // Compute attention scores: how relevant is each past token?
                // Divide by sqrt(headDim) to keep scores from getting too large,
                // which would make softmax output near-one-hot (too confident).
                var attnLogits = new Value[kH.Count];
                double scale = Math.Sqrt(headDim);
                for (int t = 0; t < kH.Count; t++)
                {
                    Value dot = new Value(0);
                    for (int j = 0; j < headDim; j++)
                        dot = dot + qH[j] * kH[t][j];
                    attnLogits[t] = dot / scale;
                }

                // Turn scores into probabilities (must sum to 1)
                var attnWeights = Softmax(attnLogits);

                // Weighted blend of value vectors: gather info from relevant tokens
                for (int j = 0; j < headDim; j++)
                {
                    Value sum = new Value(0);
                    for (int t = 0; t < vH.Count; t++)
                        sum = sum + attnWeights[t] * vH[t][j];
                    xAttn.Add(sum);
                }
            }

            // Combine all heads back together and project to output dimension
            x = Linear(xAttn.ToArray(), stateDict[$"layer{li}.attn_wo"]);

            // Residual connection: add back the original input.
            // This lets information bypass the attention layer if needed,
            // and helps gradients flow during training.
            x = x.Zip(xResidual, (a, b) => a + b).ToArray();

            // =================================================================
            // MLP (FEED-FORWARD NETWORK)
            // =================================================================
            //
            // After attention has gathered information from other tokens,
            // the MLP processes it further. Think of attention as "what info
            // do I need?" and the MLP as "what do I do with that info?"
            //
            // Structure: expand to 4x width → activation → compress back.
            // The expansion gives the model a wider "thinking space."
            //
            // Activation function: squared ReLU = max(0, x)^2
            // This is a modern choice (used in PaLM, etc.). The squaring
            // makes the function smoother and more selective than plain ReLU.
            // =================================================================

            xResidual = x;
            x = RMSNorm(x);
            x = Linear(x, stateDict[$"layer{li}.mlp_fc1"]);  // expand: nEmbd → 4*nEmbd
            x = x.Select(xi => xi.ReLU().Pow(2)).ToArray();   // squared ReLU activation
            x = Linear(x, stateDict[$"layer{li}.mlp_fc2"]);   // compress: 4*nEmbd → nEmbd
            x = x.Zip(xResidual, (a, b) => a + b).ToArray();  // residual connection
        }

        // Final step: convert the internal representation back to vocabulary scores.
        // "Weight tying" — we reuse the token embedding matrix (wte) here.
        // Intuition: the same matrix that maps tokens→vectors can map vectors→tokens.
        // This halves the parameters needed and often improves results.
        return Linear(x, stateDict["wte"]);
    }

    // =========================================================================
    // MAIN: DATASET, INITIALIZATION, TRAINING, AND GENERATION
    // =========================================================================

    static void Main(string[] args)
    {
        // Parse CLI arguments (e.g. --n_embd 32 --num_steps 2000)
        nEmbd = ParseArg(args, "n_embd", 16);
        nLayer = ParseArg(args, "n_layer", 1);
        blockSize = ParseArg(args, "block_size", 8);
        numSteps = ParseArg(args, "num_steps", 1000);
        nHead = ParseArg(args, "n_head", 4);
        learningRate = ParseArg(args, "learning_rate", 1e-2);
        seed = ParseArg(args, "seed", 42);
        rng = new Random(seed);
        headDim = nEmbd / nHead;

        // ---------------------------------------------------------------------
        // DATASET: Download and load a list of human names (one per line).
        // This is our training data. The model will learn the statistical
        // patterns of these names and then generate new ones.
        // ---------------------------------------------------------------------
        string inputFile = "input.txt";
        if (!File.Exists(inputFile))
        {
            Console.WriteLine("Downloading names dataset...");
            using var http = new HttpClient();
            var text = http.GetStringAsync(
                "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
            ).Result;
            File.WriteAllText(inputFile, text);
        }

        var allText = File.ReadAllText(inputFile);
        var docs = allText.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                          .Select(l => l.Trim())
                          .Where(l => l.Length > 0)
                          .ToList();

        // Shuffle so the model doesn't see names in alphabetical order
        for (int i = docs.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (docs[i], docs[j]) = (docs[j], docs[i]);
        }

        // Build the tokenizer from the dataset
        tokenizer = new Tokenizer(docs);
        Console.WriteLine($"vocab size: {tokenizer.VocabSize}, num docs: {docs.Count}");

        // ---------------------------------------------------------------------
        // MODEL PARAMETERS: Create all weight matrices.
        //
        // wte: Token Embedding table. Each row is a learned vector for one token.
        //      Shape: [vocabSize x nEmbd]. "a" gets one vector, "b" gets another, etc.
        //
        // wpe: Position Embedding table. Each row encodes a position (0, 1, 2, ...).
        //      Shape: [blockSize x nEmbd]. Position 0 gets one vector, position 1 another.
        //
        // Per layer:
        //   attn_wq, wk, wv: Project input into Query, Key, Value vectors for attention.
        //   attn_wo:         Combine attention head outputs back together.
        //   mlp_fc1:         Expand from nEmbd to 4*nEmbd (wider "thinking space").
        //   mlp_fc2:         Compress from 4*nEmbd back to nEmbd.
        //
        // Note: attn_wo and mlp_fc2 are initialized to all zeros. This means
        // at the start, the residual connections pass through unchanged — the
        // model begins as an identity function and gradually learns to do more.
        // This is a stability trick used in many modern architectures.
        // ---------------------------------------------------------------------
        stateDict["wte"] = Matrix(tokenizer.VocabSize, nEmbd);
        stateDict["wpe"] = Matrix(blockSize, nEmbd);
        for (int i = 0; i < nLayer; i++)
        {
            stateDict[$"layer{i}.attn_wq"] = Matrix(nEmbd, nEmbd);
            stateDict[$"layer{i}.attn_wk"] = Matrix(nEmbd, nEmbd);
            stateDict[$"layer{i}.attn_wv"] = Matrix(nEmbd, nEmbd);
            stateDict[$"layer{i}.attn_wo"] = Matrix(nEmbd, nEmbd, std: 0); // zero init
            stateDict[$"layer{i}.mlp_fc1"] = Matrix(4 * nEmbd, nEmbd);
            stateDict[$"layer{i}.mlp_fc2"] = Matrix(nEmbd, 4 * nEmbd, std: 0); // zero init
        }

        // Collect all parameters in a deterministic order.
        // The optimizer tracks per-parameter state (momentum, etc.), so the
        // order must be stable across iterations.
        var paramKeys = new List<string> { "wte", "wpe" };
        for (int i = 0; i < nLayer; i++)
        {
            paramKeys.Add($"layer{i}.attn_wq");
            paramKeys.Add($"layer{i}.attn_wk");
            paramKeys.Add($"layer{i}.attn_wv");
            paramKeys.Add($"layer{i}.attn_wo");
            paramKeys.Add($"layer{i}.mlp_fc1");
            paramKeys.Add($"layer{i}.mlp_fc2");
        }
        parameters = paramKeys
            .SelectMany(key => stateDict[key].SelectMany(row => row))
            .ToList();
        Console.WriteLine($"num params: {parameters.Count}");

        Train(docs);
        Generate(numSamples: 5);
    }

    // =========================================================================
    // TRAINING
    // =========================================================================
    //
    // Each step:
    //   1. Pick a name, e.g. "Emma"
    //   2. Tokenize: [BOS, E, m, m, a, EOS]
    //   3. For each position, ask the model "what comes next?"
    //      - Given [BOS], predict E
    //      - Given [BOS, E], predict m
    //      - Given [BOS, E, m], predict m
    //      - ...and so on
    //   4. Measure how wrong it was (cross-entropy loss)
    //   5. Backpropagate to compute gradients
    //   6. Update all parameters with Adam
    //
    // The loss number tells you how well the model is doing.
    // Random guessing on 28 characters ≈ loss of ln(28) ≈ 3.33.
    // A well-trained model should get well below that.
    // =========================================================================
    static void Train(List<string> docs)
    {
        // ---------------------------------------------------------------------
        // ADAM OPTIMIZER STATE
        //
        // Adam is a smart version of gradient descent. Plain gradient descent
        // just goes: parameter -= learning_rate * gradient. Adam improves this
        // with two ideas:
        //
        // 1. Momentum (m): Instead of using just the current gradient, keep a
        //    running average. This smooths out noisy updates and helps push
        //    through flat spots. Like a ball rolling downhill — it builds speed.
        //
        // 2. Adaptive learning rate (v): Track how much each parameter's gradient
        //    varies. Parameters with consistently large gradients get smaller
        //    updates; parameters with small, noisy gradients get larger updates.
        //    Each parameter effectively gets its own tuned learning rate.
        //
        // The "hat" versions (mHat, vHat) correct for the fact that m and v
        // are biased toward zero at the start (since they're initialized to 0).
        // ---------------------------------------------------------------------
        double beta1 = 0.9, beta2 = 0.95, epsAdam = 1e-8;
        var mState = new double[parameters.Count]; // first moment (momentum)
        var vState = new double[parameters.Count]; // second moment (adaptive rate)

        for (int step = 0; step < numSteps; step++)
        {
            // Pick one training document (a name), cycling through the dataset
            var doc = docs[step % docs.Count];

            // Tokenize: convert characters to integer IDs, with BOS/EOS markers
            var tokens = tokenizer.Encode(doc);
            if (tokens.Count > blockSize)
                tokens = tokens.Take(blockSize).ToList();

            // Initialize empty KV caches (one per layer)
            var keys = Enumerable.Range(0, nLayer).Select(_ => new List<Value[]>()).ToArray();
            var values = Enumerable.Range(0, nLayer).Select(_ => new List<Value[]>()).ToArray();
            double lossF = 0.0;

            // Forward pass: process each token and predict the next one
            for (int posId = 0; posId < tokens.Count - 1; posId++)
            {
                // Run the model: input token → scores for every possible next token
                var logits = GPT(tokens[posId], posId, keys, values);

                // Convert scores to probabilities
                var probs = Softmax(logits);

                // Cross-entropy loss: -log(probability assigned to the CORRECT next token)
                // If the model gave 90% to the right answer: -log(0.9) ≈ 0.10 (low loss, good)
                // If the model gave 1% to the right answer:  -log(0.01) ≈ 4.6 (high loss, bad)
                var loss = -(probs[tokens[posId + 1]].Log());
                loss = loss * (1.0 / (tokens.Count - 1)); // average over sequence length

                // Backpropagate: compute gradients for every parameter
                loss.Backward();
                lossF += loss.Data;
            }

            // Adam optimizer: update every parameter using its gradient.
            // Linear learning rate decay: start at full LR, decrease to 0 by the end.
            // This helps the model make big jumps early and fine-tune later.
            double lrT = learningRate * (1.0 - (double)step / numSteps);
            for (int i = 0; i < parameters.Count; i++)
            {
                var p = parameters[i];

                // Update running averages of gradient (m) and squared gradient (v)
                mState[i] = beta1 * mState[i] + (1 - beta1) * p.Grad;
                vState[i] = beta2 * vState[i] + (1 - beta2) * p.Grad * p.Grad;

                // Bias correction (compensates for zero-initialization of m and v)
                double mHat = mState[i] / (1 - Math.Pow(beta1, step + 1));
                double vHat = vState[i] / (1 - Math.Pow(beta2, step + 1));

                // The actual update: nudge parameter in the direction that reduces error
                p.Data -= lrT * mHat / (Math.Sqrt(vHat) + epsAdam);

                // Reset gradient for next iteration
                p.Grad = 0;
            }

            Console.WriteLine($"step {step + 1} / {numSteps} | loss {lossF:F4}");
        }
    }

    // =========================================================================
    // INFERENCE: GENERATE NEW NAMES
    // =========================================================================
    //
    // Now the model has learned. We generate by:
    //   1. Start with BOS token
    //   2. Feed it through the model → get probabilities for next token
    //   3. Randomly pick a token (weighted by probabilities)
    //   4. Feed THAT token back in → get probabilities for the one after
    //   5. Repeat until EOS or max length
    //
    // This is called "autoregressive generation" — each output becomes
    // the next input. It's exactly how ChatGPT generates text, one token
    // at a time.
    // =========================================================================
    static void Generate(int numSamples)
    {
        Console.WriteLine("\n--- generation ---");
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            var keys = Enumerable.Range(0, nLayer).Select(_ => new List<Value[]>()).ToArray();
            var values = Enumerable.Range(0, nLayer).Select(_ => new List<Value[]>()).ToArray();
            int tokenId = tokenizer.BOS;
            var generated = new List<string>();

            for (int posId = 0; posId < blockSize; posId++)
            {
                var logits = GPT(tokenId, posId, keys, values);
                var probs = Softmax(logits);
                var weights = probs.Select(p => p.Data).ToArray();

                // Weighted random sampling: pick next token based on probabilities.
                // Higher probability tokens are more likely to be chosen, but there's
                // randomness — so the model doesn't always produce the same output.
                double r = rng.NextDouble() * weights.Sum();
                double cumulative = 0;
                tokenId = 0;
                for (int i = 0; i < weights.Length; i++)
                {
                    cumulative += weights[i];
                    if (r <= cumulative)
                    {
                        tokenId = i;
                        break;
                    }
                }

                if (tokenId == tokenizer.EOS) break;
                generated.Add(tokenizer.Decode(tokenId));
            }

            Console.WriteLine($"sample {sampleIdx}: {string.Join("", generated)}");
        }
    }
}
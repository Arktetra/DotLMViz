export const content: Record<string, string> = {
    'token-embedding':'A token embedding is a learned mapping from discrete tokens (e.g., words, subwords, or characters) in the models vocabulary to continuous high-dimensional vectors. Each token is assigned a unique vector, and these embeddings serve as the first layer of input to the Transformer.',
    'positional-embedding':'The positional embedding can be thought of as a lookup table mapping the index of the position of each token to a residual stream vector. This helps transformers to have access to positional information so they know two tokens are next to each other.',
    'layernorm':'LayerNorm is a simple normalization function applied at the start of each layer. It converts each input vector to have zero mean and unit variance, and then applies an elementwise scaling and translation.',
    'attention-head':'A component of the self-attention mechanism. Each attention head computes a weighted representation of the input sequence by focusing on different parts of the sequence, based on learned query, key, and value projections. Multiple heads allow the model to capture diverse relationships between tokens.',
    'mlp':'The MLP is a feedforward neural network in a Transformer block, with a singular hidden layer and a non-linear activation function (e.g. GeLU).',
    'unembedding':'The reverse of token embedding. It maps the final hidden representations of the decoder back to the token vocabulary space, producing logits that represent the models prediction probabilities for the next token.'

};
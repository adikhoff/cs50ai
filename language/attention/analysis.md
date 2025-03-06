# Analysis

## Layer 8, Head 10
As evidenced in Attention_Layer8_Head10.png, [MASK] pays attention
to the appropriate verb, and proceeds to guess tobacco products in
the first sentence. The weakness is that it doesn't always take the
entire context, so in the second sentence it assumes that 'consumed'
refers to consumable items.

Example Sentences:
- Holmes proceeded to smoke his large red [MASK].
- It was late at night and he was low on nicotine. So he consumed a [MASK].

## Layer 1, Head 4
Attention_Layer1_Head4.png shows that [MASK] pays attention to the
preceding two words to conclude that it must be an animal.
In the second example, it uses the preceding two words to guess
that it must be a geographical feature.

Example Sentences:
- The quick brown [MASK] jumps over the lazy dog.
- The rain in Spain falls mainly on the [MASK].


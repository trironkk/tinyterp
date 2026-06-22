# Let's build GPT: from scratch, in code, spelled out (ng-video-lecture)

> Source: https://github.com/karpathy/ng-video-lecture (video: https://www.youtube.com/watch?v=kCc8FmEb1nY)
> Collected: 2026-06-22
> Published: 2023-01-17

_Note: Captured from the repository README via GitHub; condensed faithfully, not verbatim._

## What it is

This repository (`nanogpt-lecture`) holds the code created in the
[Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series,
specifically the lecture **"Let's build GPT: from scratch, in code, spelled out."** It is
published as a standalone GitHub repo "so people can easily hack it, walk through the `git log`
history of it, etc." The companion video builds a Generatively Pretrained Transformer (a
character-level GPT trained on tiny Shakespeare) line by line.

## Author's note on initialization

> "Sadly I did not go too much into model initialization in the video lecture, but it is quite
> important for good performance. The current code will train and work fine, but its convergence
> is slower because it starts off in a not great spot in the weight space."

Karpathy points to [nanoGPT's `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py)
for the `# init all weights` comment and the `_init_weights` function. He notes the code in this
lecture repo differs from nanoGPT in how it names and stores modules, so it can't be directly
copy-pasted; the repo is kept "almost exactly what we actually covered in the video."

## Relationship to the broader course

This lecture corresponds to the "Building GPT from Scratch" lecture of the Neural Networks: Zero
to Hero course — the standalone code companion to that specific video, distinct from but
overlapping the full course repository.

## License

MIT

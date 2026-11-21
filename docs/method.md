# Method

How a picture is made, judged, and scored. The wording quoted here is in `paintbench/prompts.py`, and tests pin it.

## What the model is given

Every attempt is a fresh conversation with a system prompt, one task message, and one tool. The model gets no image, cannot look anything up, cannot run code, and never sees its own picture.

The system prompt:

```text
You are taking part in a memory test. You will be asked to type a small picture of a well-known painting from memory, one character per cell.

You have one tool, write_rows. It stores the palette and the rows you type, and it checks only their format. You have no other tools. You cannot look anything up, run code, or see your picture.

Type every row yourself, cell by cell. Do not describe the picture in place of typing it.
```

The task message, here for The Great Wave at 32 cells wide:

```text
Painting: Under the Wave off Kanagawa (often called The Great Wave), by Katsushika Hokusai, about 1831.

Type it as a grid 32 cells wide and 22 cells tall. Row 1 is the top of the picture. The first character of a row is the left edge.

How to write it
- Choose your own palette, with as many colours as you want. Each colour has a one-character code (any character except a space, # or =) and red, green and blue values from 0 to 255.
- Write each palette entry as the code, then red, green and blue, separated by spaces, for example: k 20 40 90. Put the entries in the palette field of write_rows, with your first rows or before them. You can add colours in a later call. If you send a code that is already defined, it is redefined: every cell typed with that code takes the new colour, in rows already stored too.
- Send the palette and the rows with write_rows. Each row must be exactly 32 characters long and use only your palette codes.
- You may put spaces inside a row to help you count, for example between blocks of eight. Spaces are ignored.
- Send at most 16 rows per call. Keep calling write_rows until all 22 rows are stored.
- If a row is rejected, correct it and send it again.

How the work goes
- You do not have to finish in one turn. After each write_rows call you get its result back and carry on from there, for up to 40 turns in all.
- Work a strip at a time: type a strip of rows, send it, then go on to the next. Do not work out every row of the picture before you send the first strip.

Aim
Make the picture as close as you can to the real painting: what is in it, where each thing is, its shape, its size and its colours.
```

The first line comes from the painting's `meta.yaml`. Works that exist in several versions name theirs (The Scream, Water Lilies, Sunflowers); the others give title, artist, and date only. A painting's grid height at each width is fixed in the same file, from the shape of its reference image.

The tool, `write_rows`, takes palette entries, a start row, and rows. It stores a row of the right length whose characters are all in the palette, and rejects any other. Its replies are fixed templates that hold only counts, row numbers, or one echoed character, so nothing in them says anything about the picture.

A second encoding, `value`, has no palette: every cell is typed as six hex digits. The published run uses the palette encoding.

## An attempt

The harness calls the model until every row is stored. If a reply ends before that, the model is told that the picture is not finished, at most three times. The published run allowed 40 model calls and 180 minutes per attempt.

An attempt ends in one of four states: complete (every row stored), incomplete (a limit was reached), refused (the model declined, or the provider blocked the request), or error (the API failed after retries). Only complete pictures are judged. A blocked request is sent again only when asked for (`--retry-refused`); the published run sent each one again up to twice.

## Judging

A set is every complete picture of one painting at one width. A judge gets one message: the real painting, labelled REFERENCE, then the pictures, each drawn as solid squares at the same display size, under random letters and in random order. The letters and the order are drawn afresh for every call. The judge is asked to rank them "the way a person who knows the painting well would judge them", taking into account what is in the picture, where each thing is, its shape, and the colours, with no ties.

The reply must follow a fixed form, and it is parsed strictly. A reply that does not parse is asked for once more, and is never guessed at.

The published run has two judges, Claude Fable 5.2 and Claude Opus 5.5, with three calls each per set.

A second kind of call ranks one model's own pictures of a painting, at every width, against each other ("peak size").

## Scoring

- **Rank score.** For a model in a set: the share of its comparisons with other models' pictures that it won, pooled over judges and repeats. With one picture per model this is its mean rank, scaled so that 1 is always first and 0 always last.
- **Standings at each width.** A model's figure at a width is its comparisons won over its comparisons made, pooled over the sets at that width. Each figure has a 95% interval from resampling the paintings, and two neighbours whose order the resampling does not separate share a place. The interval says how far a figure depends on the choice of paintings. It does not include the variation between attempts.
- **No overall rating.** Larger grids separate the models more, and a model's standing can change with the size, so the largest width leads. The mean over all sizes is kept as a secondary table.
- **A missing picture** counts as nothing for its model, and is not a loss. The set still counts for the others.
- **Entries kept apart.** An entry named under `aside` in the config has its pictures taken out of every ranking before scoring. The published run keeps Haiku 5.5 at its default effort apart, since the other entries ran at their highest.
- **Agreement.** Kendall's tau between judges, between a judge's own repeats, and between each judge and each person who ranked a sheet.

Formulas that compare pixels with the reference are not used.

## How far to trust the judges

A person ranked twelve blind sheets of seven pictures, from two earlier runs of Claude models only. Kendall's tau between their order and a judge's averaged order was 0.83 on average, the same for both judges. Scored from the person's rankings alone, the models came out in the same order as from the judges. On this evidence (one person, twelve sheets, Claude pictures only) the judges track a person's order of models over many sets. It does not support the order of close neighbours within one set.

No person has yet ranked sheets from the published run, which adds pictures by GPT and Gemini models. Both judges are Claude models, and with no judge from another family, whether they favour Claude pictures cannot be measured here.

In the published run the two judges' averaged orders of a set agree at tau 0.88, about as well as one judge's own repeats agree with each other (0.85). Each judge scores its own model within 0.01 of what the other judge gives it.

## Known limits

- One attempt per model, painting, and width.
- A rank score is relative to the other pictures in the same run. Figures from two runs cannot be compared.
- Some requests are blocked by the provider. In the published run eleven attempts, all by Claude models at 128 wide, were blocked on each of three tries and have no picture, which is why some models are scored on fewer than sixteen paintings at that width. Claude Opus 5 was blocked on every request, so it is not in the run.
- A complete picture is not always a detailed one: at large sizes weaker models often repeat rows, or type a coarse picture enlarged.
- The cost estimate assumes far fewer thinking tokens than the newest models spend at their highest effort. Its counts of attempts and judging calls are exact.
- Models without tool calling are not supported.

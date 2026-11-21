"""The score stage: read a run directory and compute everything the report shows.

It calls no model and `compute_scores` writes nothing. It reads the attempt
stats, the judgement records and the human rankings (shapes in layout.py).

The rules
---------
Rank score. A model's rank score in one set (one painting, one width) is the
share of its comparisons against OTHER models' pictures that it won, pooled
over its attempts and the judging calls. A ranking of n pictures is n(n-1)/2
comparisons: the picture at rank r beats the n - r below it. A pairwise call
is one comparison. With one attempt per model, and every call ranking the
same n pictures, this is exactly the mean rank across judge repeats, scaled
so that 1 means always best and 0 always worst: (n - mean rank) / (n - 1).
Two attempts of the same model are never compared for the score, otherwise a
model with several attempts could not reach 1.

Overall. A model's overall rank score is the mean of its rank scores over
every set that scored it, each set counting once. A set in which some model
has no complete picture still counts for the models it scored. For the model
without a picture it counts as nothing: a missing picture is not a loss. So
two models' overall figures can be means over different sets. `sets_scored`
says how many sets a model's figure rests on, and a model scored in fewer sets
than the list holds is marked (`coverage_differs`): its figure leaves out the
sets it has no picture in, and in those sets the other models met one rival
fewer. The like-for-like figure is kept beside the overall one
(`rank_score_common_sets`): the same mean over the common sets only. The
common sets are those in which every model that has a rank score anywhere has
one, so that figure never compares models on different sets. With no common
set there is no such figure. A judge's own figures are worked the same way
over that judge's sets.

Partial schedule. A set has a partial schedule when some two attempts of
different models were never compared (judge.pairs_per_attempt leaves pairs
out; so do rankings made on different pictures). The rank score is still wins
over comparisons, so in such a set it also depends on which opponents each
attempt met. The Bradley-Terry order allows for how strong those opponents
were, and is the one to read there.

Order. Within a set the attempts are ordered best first: by mean rank across
the judging calls when the set holds only rankings (mode "ranking"), by
Bradley-Terry strength when it holds any pairwise call (mode "pairwise"; a
ranking in such a set is broken into its comparisons and feeds the same fit).
Equal mean ranks or strengths are ties; tied attempts are listed in attempt-id
order. Strengths are on a log scale: a picture d above another is preferred
with probability 1 / (1 + e^-d). Every picture is given BT_PRIOR wins and
losses against a notional opponent, whose strength is the zero of the scale,
so a picture that never lost (or never won) stays finite.

Pooling. "Combined over judges" always means every judge's calls pooled, each
call counting once. It is not a mean of per-judge figures.

Failed judgements. A record is used only if its status is "ok", it ranks at
least two pictures, it is not superseded and it is not stale. Every other
record is counted and listed, never scored. `is_used` is that rule, and the
report goes by it too. A human ranking is used once it has been recorded
(`is_recorded`).

Stale judgements. A record that says which pictures it was shown ("pictures",
see layout.py) is stale once any of those attempts is no longer complete or
holds a different picture: what it ranked is no longer in the run. A record
that says which reference image the judge was shown ("reference", the
SHA-256 of the file) is stale once the painting's reference file is another:
it was ranked against a different picture of the painting. Either way it is
listed with status "stale". A record with neither key (imported, or made
before the key existed) is never stale on that account. When a painting's
metadata or reference file cannot be found, the reference check is skipped
for its records and the painting is named in `reference_unchecked`.

A record made before its set grew still ranks the pictures it shows, so it is
used. The judge stage makes the call again if it still plans it, and the new
record then takes the old one's place; the complete attempts that no record
in use ranks are listed with their grid (`not_judged`).

Superseded judgements. The judge stage judges a set whole or in pairs, never
both: in pairs once the set's complete attempts number more than
judge.max_per_call. A ranking left from before a set grew past that size, or
a pair left from before it shrank back, is a call the stage no longer plans,
so it is never replaced; pooled with the current calls it would count some
pairs twice. Such a record is not used and is listed with status
"superseded"; its file stays on disk. The rule needs the config. Without one
it is not applied, and a set that holds both kinds pools them.

Unknown is not zero. A null stats field (imported attempts have many) is left
out of means and totals; `known` says how many attempts reported each field.
The four care figures (rows rejected on first submission, rows rejected,
nudges, replies cut off) are averaged over complete and incomplete attempts
only: an attempt that refused or errored says nothing about care.

Nudges. A nudge is the reminder sent when a reply makes no tool call and rows
are missing. The loop sends it whether the model stopped early or the output
limit cut its reply off, and only the first says anything about care. So the
"nudges" figure counts the nudges that did NOT follow a cut-off reply, and the
cut-off replies have a figure of their own (`truncated_replies`). How many
nudges followed a cut-off is the attempt's "nudges_after_cutoff"; for an
attempt saved without it the number is worked out from what is on disk
(`nudges_after_cutoff`), and where that cannot be done the nudges figure is
unknown.

Refusals. The state "refused" covers two different things, told apart by the
attempt's "refusal_detected_by": "provider" is a platform block (the provider
stopped the request; the model did not decline), "text" is the model declining
in words. The counts keep them apart, and no output words a provider's block
as the model declining.

Agreement. Two orders are compared over the attempts both contain. Between
strict orders the figure is Kendall's tau; where one side is a combined order,
which can hold ties, it is tau-b. A swapped pair is a pair the two put in
opposite order (a tie is not a swap). "Within one adjacent swap" means
identical, or differing by the exchange of one pair of neighbours; for a
combined order, ties are broken in the other side's favour first, so it means
at most one swapped pair.

Planned and run. With a config, the planned grid is what the run's saved
config asks for (cfg.attempt_jobs()). A planned attempt with no stats.json was
never started, or was stopped before its record was written; the run keeps no
reason either way. Without a config the plan is not known (`planned` is null).

Intervals. The interval on an overall rank score is a 95% percentile interval
from resampling paintings. The paintings with a set that scored some model are
drawn with replacement, as many as there are, INTERVAL_DRAWS times from a
fixed seed, so the same run always gives the same figures. Each time the
figure is worked again: the mean of the model's rank scores over every set of
the paintings drawn that scored it, a painting drawn twice counting twice. The
interval runs from the 2.5th to the 97.5th percentile of those figures. The
difference between two models is resampled the same way, the same paintings
being drawn for both and each model's mean being taken over the sets that
scored it, and their order is "separated" when the interval of the difference
does not include zero. A difference is rounded to 12 places, in every
resample and as it is given, so that two figures that are equal differ by
exactly zero: unrounded, a tie would come out just above or just below zero
by the chance of floating-point rounding, and would be counted as an order or
as its reverse. The interval says how far a figure depends on which paintings
were chosen. It does not include the variation between two attempts by one
model at one painting and width: with one attempt per model and set that is
not measured, so the intervals are if anything too narrow. With fewer than
two paintings there is nothing to resample and no interval. A model with
scores at only one painting has no interval either, and nor has a difference
it is part of: for such a difference there is too little to tell, and its
order is neither separated nor shown to be level (`separated` is null). The
two models share a place all the same, as two whose order is not separated
do: a place of its own would say that the order is established. A resample
whose paintings hold no set that scored the model gives no figure and is left
out of the percentiles.

By width. A model's rank score at one width is its wins over its comparisons,
pooled over every set at that width that scored it and over judges. The entry
says how many sets that is. Like every rank score it is a standing against the
other models at that width, not a measure of the pictures themselves.

Standings. Every width with a judged set has standings of its own
(`standings_by_width`): the models with a figure there, in order of that
figure (equal figures in name order), each with an interval, and each two
neighbours held against each other on those figures. The paintings resampled
are those with a set at that width that scored some model (`paintings`), as
many as there are and the same ones for every model: where a width was
judged on fewer paintings than the run holds, the others are not drawn. In
each resample each model's wins and comparisons are summed again over
the sets at that width, of the paintings drawn, that scored it. A model's
interval is that of its pooled figure, and a difference is that of two pooled
figures, so the difference, its interval and whether the order is separated
are all about the figures the entry shows. (Between two neighbours of the
overall order the difference is one of two means of set scores, each set
counting once, because the overall figure is such a mean.) Where the interval
of a difference includes zero the order of the two is not separated: as far
as the run can tell, they are level at that width. Places follow as they do
overall: only a separated pair starts a new place, and a pair with no
interval shares one. A model with no figure at a width is not in its
standings. The widths are rated apart, and the largest that has a judged set
is the one a summary leads with (`default_width`): larger sizes tell the
models apart more, so the mean over all sizes is kept as a secondary figure
and is not the headline. `headline.top_by_width` is the first pair of each
width's standings.

Distinct rows. A picture is complete when every row is stored, whatever the
rows hold. The share of distinct rows is the number of different rows, as
they are drawn, over the number of rows: two rows typed with different codes
for the same colours are one row. A picture finished by repeating a row, or
by typing a coarse grid enlarged, has a low share.

Own model. A judge whose model id (judge_settings.model) is also a
contestant's is compared with the other judges on that contestant: its rank
score from this judge against its rank score from the others, over every set
in which this judge and at least one other judge scored it. The comparison is
paired set by set, whether or not every model has a picture there, and this
judge's score is held against the mean of the others'. The judges are
matched on the painting and the width, not on what they ranked the model
against: where they were not given the same pictures in a set (a judge
whose rankings are not all in, say), that is part of the difference too. Its two scores are means over those sets. A
set that only one judge scored the contestant in is in neither, so they can
differ from the judges' overall figures. The difference cannot say which
judge is off.

Agreement between judges is given at three levels: between the judges' orders
of a set, each averaged over its repeats (tau-b, as in `judge_vs_judge`);
between one ranking call of one judge and one of another; and between two
ranking calls of the same judge. The last two are Kendall's tau over the
pictures both calls ranked, averaged over every such pair of calls.

Untrusted text. `refusal_text`, `error` and `parse_errors` (in `not_complete`,
`retried` and `failed_judgements`) hold text a model wrote or caused. They are
passed through as data; escape them wherever they reach HTML. `format_scores`
never prints them.

Output shape
------------
`compute_scores` returns plain JSON data (string keys, lists, numbers,
strings, booleans, null), so scores.json read back is equal to what was
returned. Every key below is always present. "id" means an attempt id,
"<painting>/<width>/<model>/<n>". A pair is a two-item list of ids.

    scores_version     7
    harness_version    str
    run_id             str, the run directory's name
    counts             {attempts, judgements, judgements_failed, judgements_stale, judgements_superseded,
                       human_rankings, human_rankings_pending, human_rankings_stale}
                       attempts = the attempts with a record on disk (see `planned` for the planned grid)
                       judgements = records used; judgements_superseded = records of a kind their set is no
                       longer judged by; judgements_stale = records that would otherwise be used but are
                       stale; judgements_failed = the other records not used; human_rankings = recorded ones
                       human_rankings_stale = recorded rankings of pictures that have since changed (left out)
    aside              [model]          the entries the config keeps apart: their pictures are taken out of every
                                        ranking before it is scored, so they have no rank score and are in no other's
    planned            PLANNED | null   the planned grid against what was run; null without a config
    grids              [GRID]     one per painting and width with an attempt, a judgement or a human
                                  ranking; sorted by painting, width
    sets               [SET]      one per painting, width and judge; sorted by painting, width, judge
    combined           [SET]      one per painting and width, all judges' calls pooled; sorted likewise
    models             [MODEL]    one per contestant model; best overall rank score first, unscored last,
                                  then name
    headline           HEADLINE   how firmly the order of `models` is established
    default_width      int | null, the largest width in `standings_by_width`: the standings a summary leads with
    standings_by_width [STANDINGS] one per width at which a `combined` set scored some model, widths ascending
    peak               [PEAK]     one per painting, model and judge; sorted by painting, model, judge
    peak_combined      [PEAK]     one per painting and model, all judges' calls pooled
    peak_tally         [PEAK_TALLY]  one per model in `peak_combined`, sorted by model
    pictures           [PICTURE]  one per complete attempt whose picture.txt can be read; sorted by painting,
                                  width, model, n
    pictures_by_width  [{model, width, pictures, under_half, mean_share_distinct}] one per model and width in
                                  `pictures`, sorted by model, width. under_half = the pictures with fewer than
                                  half their rows distinct; mean_share_distinct is over the pictures
    completion         {total: COMPLETION, by_model: [COMPLETION], by_cell: [COMPLETION],
                        by_model_width: [COMPLETION], by_width: [COMPLETION]}
                       by_model sorted by model; by_cell by model, painting, width; by_model_width by model,
                       width; by_width by width
    not_complete       [NOT_COMPLETE]    every attempt whose state is not "complete"; sorted by painting,
                                         width, model, n
    retried            [RETRIED]         every attempt that replaced an earlier run of itself (its stats hold
                                         "earlier"), whatever its state now; sorted likewise
    judge_vs_person    [JUDGE_VS_PERSON] one per recorded human ranking and judge with a set for the same
                                         painting and width, where both ranked at least two of the same attempts
    judge_vs_judge     [JUDGE_VS_JUDGE]  one per pair of judges with sets for the same painting and width
    own_family         [OWN_FAMILY]      one per judge with a set, sorted by judge
    own_model          [OWN_MODEL]       one per judge and contestant with the same model id, sorted by judge,
                                         contestant
    judges             [JUDGE]           one per judge with a record in use, sorted by judge
    judge_agreement    JUDGE_AGREEMENT   how far the judges agree with each other and with themselves
    failed_judgements  [FAILED]          every judgement record not used: failed, superseded or stale
    reference_unchecked [str]            the paintings with a judgement or a person's ranking that names its
                                         reference image, where the image could not be found to compare

PLANNED
    attempts           int, the attempts the config plans
    run                int, those of them with a record (a stats.json that can be read) on disk
    not_started        [{attempt, painting, width, model, n}] the planned attempts with no stats.json, sorted
                       by painting, width, model, n. No reason is recorded for any of them. (A record that is
                       on disk and cannot be read is in neither figure.)
    unplanned          int, the attempts on disk that the config does not plan (imported, or planned by an
                       earlier config)
    by_model           [{model, planned, run, not_started}] one per model with a planned attempt, by model
    by_model_width     [{model, width, planned, run, not_started}] by model, width
    by_width           [{width, planned, run, not_started}] by width

GRID
    painting, width    str, int
    height             int | null
    complete           [id]   the complete attempts on disk, judged or not; sorted by model, n
    not_judged         [id]   those of `complete` that no ranking or pairwise record in use ranks. With two or
                              more complete attempts here, running the judge stage again ranks them
    judges             [str]  judges with a set here
    people             [str]  people with a recorded ranking here

SET (an entry of `sets` or `combined`)
    painting, width    str, int
    height             int | null
    judge              str; null in `combined`
    judges             [str]  the judges whose calls are in the set (one name in `sets`)
    mode               "ranking" | "pairwise"
    calls              int, the judgement records used
    common             bool, a common set: every model with a rank score in any set of the same list (`combined`,
                       or this judge's entries of `sets`) has one here. The common-sets figures use these sets
                       only. The overall figures use every set
    partial_schedule   bool, some two attempts of different models in the set were never compared: rank
                       scores here depend on which opponents each attempt met
    order              [ATTEMPT] best first
    models             [SET_MODEL] best rank score first, unscored last, then name
  ATTEMPT
    attempt            id
    model, n           str, int
    mean_rank          float (1 = best), over the calls that ranked it; null in pairwise mode
    ranks              [int] its rank in each of those calls, in judge then repeat order; [] in pairwise mode
    strength           float, Bradley-Terry strength on the log scale described above (higher is
                       better, 0 = the notional opponent), rounded to 6 places; null in ranking mode
  SET_MODEL
    model              str
    family             str | null
    attempts           int, this model's attempts in the set
    rank_score         float 0..1; null if it met no other model's picture
    wins, comparisons  int, int   rank_score = wins / comparisons
    mean_rank          float, the mean of all its attempts' ranks; null in pairwise mode

MODEL
    model              str
    family             str | null
    rank_score         float | null, the mean of its rank scores in the sets of `combined` that scored it, each
                       set counting once; null if no set scored it
    sets_scored        int, how many `combined` sets scored it: the sets rank_score is a mean over
    sets_all           int, how many `combined` sets scored some model (the same in every entry)
    coverage_differs   bool, sets_scored != sets_all: some set that scored other models did not score it, so
                       its figure is a mean over fewer sets than the list holds
    rank_score_common_sets  float | null, the like-for-like figure: the mean of its rank scores in the common
                       sets of `combined` only; null if it has no score or there is no common set
    sets_common        int, how many `combined` sets are common (the same in every entry)
    rank_score_interval  [low, high] | null, the interval on rank_score from resampling paintings (see
                       Intervals); null with no rank score, or with scores at fewer than two paintings
    by_width           [{width, sets, rank_score, wins, comparisons}] one per width at which a `combined` set
                       scored some model: wins over comparisons pooled over the `sets` sets at that width that
                       scored it, a number that can differ from model to model; rank_score null, and sets 0,
                       if none did
    by_judge           [{judge, rank_score, sets_scored, sets_all, coverage_differs, rank_score_common_sets,
                       sets_common}] the same from each judge's own sets; judges that scored it only, sorted
                       by judge

HEADLINE
    draws, seed        int, int: the resamples behind every interval, and the seed they are drawn from
    level              0.95
    paintings          int, the paintings with a `combined` set that scored some model: what is resampled
    sets               int, the `combined` sets that scored some model
    sets_common        int, the common sets among them
    sets_by_width      [{width, sets, sets_common}] the same two counts by width; one per width in `by_width`
    top_by_width       [{width, higher, lower, higher_score, lower_score, difference, interval, separated,
                       share_reversed}] one per width at which two models have a rank score: the first two
                       of that width's standings with their figures (higher_score and lower_score), and the
                       last four keys of the first of its pairs
    pairs              [PAIR] one per two neighbouring entries of `models` that both have a rank score
    tiers              [[model]] `models` with a rank score, in order, split wherever two neighbours are
                       separated (`separated` true) and nowhere else. The models of one tier are joined by
                       pairs whose order is not separated, or that have no interval to tell by: `pairs` says
                       which
  PAIR
    higher, lower      str, str: the model listed first, and the one after it
    difference         float, rank_score of `higher` minus rank_score of `lower`, each a mean over the sets
                       that scored that model; rounded to 12 places, so that equal figures give exactly 0
    interval           [low, high] | null, the interval on the difference (the same paintings for both); null
                       where either model has scores at fewer than two paintings
    separated          bool | null: the interval does not include zero; null without an interval, where there
                       is too little to tell: that is neither a separated order nor a level one
    share_reversed     float | null, the share of resamples in which the difference was below zero. A
                       resample in which the two figures are equal is not one of them
    by_judge           [{judge, difference, interval, separated}] the same from each judge's calls alone, each
                       model over the sets in which that judge scored it; judges that gave both an overall
                       figure only
    reversed_by        [judge] the judges on whose calls alone `lower` is ahead

STANDINGS (the models at one width, in order of their figure there)
    width              int
    sets               int, the `combined` sets at that width that scored some model
    paintings          int, the paintings of those sets: what is resampled
    models             [{model, rank_score, interval, wins, comparisons, sets_scored, place}] the models with a
                       rank score at that width, best first, equal figures in name order. rank_score, wins and
                       comparisons are those of the model's by_width entry, and sets_scored is its `sets`.
                       interval is [low, high] | null: the interval on that pooled figure from resampling
                       paintings, null where the model has scores at fewer than two paintings there. place is
                       '1', or '2-3' for each of the neighbours whose order is not separated or has no
                       interval to tell by (`tier_places`)
    pairs              [{higher, lower, difference, interval, separated, share_reversed}] one per two
                       neighbours of `models`, as in PAIR, but of the pooled figures: `difference` is the
                       rank_score of `higher` minus that of `lower`, and in each resample both are worked
                       again as wins over comparisons, pooled over the sets at that width, of the paintings
                       drawn, that scored the model
    tiers              [[model]] `models`, in order, split wherever two neighbours are separated and nowhere else

PEAK (from the "peak" judgements: one model's sizes ranked against each other)
    painting, model    str, str
    judge              str; null in `peak_combined`
    judges             [str]
    calls              int
    widths             [{width, mean_rank, ranked}] by width; mean_rank is over the `ranked` times a
                       picture of that width was ranked
    best_widths        [int] every width with the lowest mean rank
    best_width         int; null when tied
    tied               bool, true when best_widths has more than one width
    widths_not_complete  [int] the widths at which the model has no complete picture of the painting, of the
                       widths it was planned at (with a config) or has a record at
    larger_not_complete  [int] those of them above every width in best_widths. Where there are any, the best
                       width is the best of the sizes the model finished, and cannot be read as a peak

PEAK_TALLY (the rows of `peak_combined` counted by model)
    model              str
    paintings          int, its rows
    outright           [{width, paintings}] one per width that any row ranks: the paintings on which that
                       width alone was judged best
    tied               [{widths: [int], paintings}] the paintings on which those widths tied for best
    larger_not_complete  int, its rows in which a larger width has no complete picture

PICTURE
    attempt, painting, width, model, n  id, str, int, str, int
    rows               int, the rows of the picture
    distinct_rows      int, how many different rows it holds, as drawn
    share_distinct     float, distinct_rows / rows

COMPLETION (`total` has no model; `by_model` has model and family; `by_cell` adds painting and width)
    model, family, painting, width      as above, where they apply
    attempts                            int
    complete, incomplete, refused, error  int, the count in each state
    refused_by_provider, refused_in_words  int, int: those of `refused` that the provider blocked, and those
                                        the model declined in words. A refusal whose record does not say
                                        which is in neither
    refused_by_provider_after_cut_off   int, those of refused_by_provider with at least one earlier reply
                                        cut off by the output limit
    cut_off                             {state: int} the attempts in each state with at least one reply cut
                                        off by the output limit
    no_tool_call                        int, of the complete and incomplete attempts (those the care figures
                                        cover), how many made no tool call, so had no row to be rejected
    share_complete                      float; null when there are no attempts
    mean_rows_rejected_first_submission, mean_rows_rejected, mean_nudges, mean_truncated_replies
                                        float | null, over complete and incomplete attempts. Nudges are
                                        those that did not follow a reply cut off by the output limit;
                                        truncated replies are the replies so cut off
    mean_model_calls                    float | null (this and all below: over attempts in any state)
    input_tokens, output_tokens, reasoning_tokens              int | null, totals
    mean_input_tokens, mean_output_tokens, mean_reasoning_tokens  float | null
    cost_usd                            float | null, total
    mean_wall_seconds                   float | null
    known                               {field: int} how many attempts reported each stats field behind
                                        the figures above: rows_rejected_first_submission, rows_rejected,
                                        nudges, truncated_replies, model_calls, input_tokens, output_tokens,
                                        reasoning_tokens, cost_usd, wall_seconds
    A mean or total covers the attempts that reported the field, and is null when none did. A cost total
    leaves out the attempts of a model with no price: known.cost_usd says how many it covers.
    `by_model_width` has model, family and width; `by_width` has width.

NOT_COMPLETE
    attempt, painting, width, model, n  id, str, int, str, int
    + the ENDING keys

RETRIED
    attempt, painting, width, model, n  id, str, int, str, int
    state, stop                         as the attempt stands now
    earlier                             [ENDING + {finished_at: str | null}] the runs it replaced, oldest first

ENDING (how one run of an attempt ended)
    state                               "incomplete" | "refused" | "error" (or null if stats has none); in
                                        `retried` also "complete"
    stop                                str | null, why the loop ended
    refusal_detected_by                 "provider" (a platform block) | "text" (the model declined in words) |
                                        null (not a refusal, or the record does not say)
    refusal_category                    str | null, the provider's category for its block, from the saved
                                        response; null for a run whose transcript was replaced
    every_reply_cut_off                 bool, it made at least one model call and the output limit cut off
                                        every reply
    model_calls, truncated_replies      int | null, its model calls, and the replies among them that the
                                        output limit cut off. A provider's block can follow cut-off replies
    refusal_text, error                 str | null (untrusted)

JUDGE_VS_PERSON
    painting, width, person, judge      str, int, str, str
    mode                                the judge's set mode
    person_order                        [id] the person's ranking, best first
    repeats                             [REPEAT] one per ranking call of the judge that shares at least
                                        two attempts with the person, in repeat order; [] if the judge
                                        made only pairwise calls
    repeats_compared                    int, len(repeats)
    repeats_identical                   int
    repeats_within_one_adjacent_swap    int (the test a judge is held to: at least two of three)
    combined                            AGREEMENT between the person's order and the judge's set order
  REPEAT
    repeat             int | null
    judge_order        [id] that call's ranking, best first
    n_common           int
    tau                float | null, Kendall's tau
    swapped_pairs      [pair], each with the attempt the person put higher first
    identical          bool, the same order over the attempts both ranked
    within_one_adjacent_swap  bool

JUDGE_VS_JUDGE
    painting, width, judge_a, judge_b   judge_a sorts before judge_b
    + the AGREEMENT keys, between the two judges' set orders

AGREEMENT
    n_common           int, attempts in both orders
    tau_b              float | null (null if fewer than two in common, or one side ties everything)
    swapped_pairs      [pair], each with the attempt the first side (the person, or judge_a) put higher first
    tied_pairs         [pair] pairs that either side ties
    within_one_adjacent_swap  bool, at most one swapped pair

OWN_FAMILY (do judges favour their own model family?)
    judge              str
    family             str | null, from the judge's judge_settings.family
    own_models         [str] contestants of that family that the figures cover
    other_judges       [str] the judges it is compared with: those of another family. A judge with no
                       family recorded counts as one, so record every judge's family
    same_family_judges [str] the other judges of its own family that have a set. They are kept out of the
                       comparison: a judge that may share the favour is no baseline for it
    scores_compared    int, the (painting, width, own model) rank scores that this judge and at least
                       one of `other_judges` both gave
    own_mean           float | null, the mean of this judge's rank scores over those
    others_mean        float | null, the mean of the `other_judges`' (averaged per score first)
    difference         float | null, own_mean - others_mean; positive = kinder to its own family
    Sets in which every scored contestant is of the judge's family are left out: with no outsider
    to rank them against, their mean score says nothing about favour. The three figures are null
    when there is no family, no judge of another family, or nothing left to compare.

OWN_MODEL (a judge that is also a contestant, on that contestant's pictures)
    judge, contestant  str, str: the judge, and the contestant model with the judge's model id
    model_id           str, the model id they share
    other_judges       [str] the judges it is compared with: every other judge that scored the contestant
                       in one of the sets
    sets               int, the sets in which this judge and at least one other judge gave the contestant a
                       rank score, whether or not every model has a score there
    own_score          float | null, the mean of this judge's rank scores for the contestant over those sets
    others_score       float | null, the mean of the other judges' (averaged per set first)
    difference         float | null, own_score - others_score; positive = this judge scored it higher
    interval           [low, high] | null, the interval on the difference from resampling paintings
    sets_higher, sets_lower, sets_equal  int: the sets in which this judge's score was above, below and
                       equal to the others'
    The four figures are null when there is no other judge to compare with.

JUDGE
    judge              str
    model              str | null, the judge's model id, from its judge_settings (else the config)
    family             str | null
    contestants        [str] the contestant models with the same model id: the judge is itself a contestant
    calls              int, its records in use (ranking, pairwise and peak)
    calls_reporting_api_retries  int, those whose usage holds an api_retries count
    calls_with_api_retries       int, those whose count is above zero
    api_retries        int, the retries of those calls, summed
    calls_without_reference      int, those that name no reference image (made before the key existed, or
                       imported): they are held to their pictures alone
    calls_with_order   int, those that say in which order the pictures were shown
    first_shown_best   int, those of calls_with_order in which the picture shown first was ranked best
    first_shown_expected  float, what a judge blind to the order would average: the sum over those calls of
                       1 / pictures
    A call that failed and was made again leaves no trace: the judge stage keeps only the record that stands.

JUDGE_AGREEMENT
    between_orders     {tau_b, sets, lowest}: the mean and the lowest tau_b of `judge_vs_judge`, and how many
                       entries have one
    between_calls      {tau, pairs, sets}: one ranking call of one judge against one of another, on the same
                       painting and width: mean Kendall's tau, the pairs of calls, and the sets they are in
    within             [{judge, tau, pairs, sets}] two ranking calls of the same judge on one set; judges with
                       at least one such pair, by judge
    A mean is null when there is nothing to average.

FAILED
    kind, painting, width, judge, model, repeat   as in the record (width null for peak, model peak only)
    status             "superseded" or "stale" for such a record, otherwise the record's status
    stale_reason       "picture" | "reference" for a stale record: what has changed since. null otherwise
    parse_errors       [str] why each reply was rejected (untrusted)
    error              str | null, the judge stage's own account of the failure, if it saved one (untrusted)
    path               the record's path relative to the run directory
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import yaml

from . import __version__
from .canvas import Picture, load_picture
from .layout import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    STATE_REFUSED,
    STATES,
    AttemptRef,
    RunLayout,
    attempt_id,
    parse_attempt_id,
    read_json,
    write_json,
)
from .paintings import load_painting
from .reference import sha256_file
from .types import STOP_MAX_TOKENS

if TYPE_CHECKING:
    from .config import RunConfig

# 2: overall figures over the common sets; stale judgements; partial schedules
# 3: superseded judgements; the reference a judgement was made against; provider blocks told from refusals in words;
#    nudges that followed a cut-off reply left out of "nudges", and the cut-off replies counted; retried attempts
# 4: the planned grid and the attempts never started; an interval on each overall rank score and the neighbouring
#    orders it leaves open; rank score by width; peak-size tallies and the widths a peak row lacks; the distinct rows
#    of each picture; judges that are contestants, a judge on its own model, agreement at three levels, API retries
# 5: the overall rank score, its interval, the figures by width, the differences between neighbours, the tiers and
#    the top two by width are taken over every set that scored the model, where version 4 took them over the common
#    sets only: a set in which some model has no complete picture now counts for the others. The common-sets overall
#    figure is kept as rank_score_common_sets. sets_all is new; coverage_differs now marks a model scored in fewer
#    sets than the list holds; a by_width entry's `sets` is the model's own count; headline.sets counts every set
#    that scored some model, with sets_common beside it. The own-model comparison is still over the common sets
# 6: the top two by width are held against each other on their pooled figures (wins over comparisons, worked again in
#    every resample), where version 5 took the difference of two means of set scores beside pooled figures; a
#    difference is rounded to 12 places, so that a tie is exactly zero and is not counted as reversed; a tier is split
#    only where two neighbours are separated, so two whose difference has no interval now share one; the own-model
#    comparison is over every set in which the judge and another judge scored the contestant, not the common sets only
# 7: standings for each width (standings_by_width): every model with a figure there, its interval, each two neighbours
#    held against each other, tiers and places; default_width, the largest width, which a summary leads with. The top
#    two by width are the first pair of those standings. Every figure of version 6 is as it was
SCORES_VERSION = 7

BT_PRIOR = 0.1  # wins, and losses, each item is given against the notional opponent
BT_TOLERANCE = 1e-12  # stop when no strength moved by more than this share of itself in a sweep
BT_MAX_SWEEPS = 10_000
_OPPONENT = object()

INTERVAL_DRAWS = 4000  # resamples of the paintings behind every interval
INTERVAL_SEED = 1795131391  # fixed, so that the same run always gives the same intervals
INTERVAL_LEVEL = 0.95
DIFFERENCE_PLACES = 12  # a difference of two figures is rounded to this many places, so that a tie is exactly zero

CARE_FIELDS = ("rows_rejected_first_submission", "rows_rejected", "nudges", "truncated_replies")
USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "reasoning_tokens", "cost_usd", "wall_seconds")
_CATEGORY = re.compile(r"[A-Za-z0-9_.:-]{1,64}")  # a provider's category is a short code; anything else is not taken for one


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------- agreement


def _positions(order: Sequence[Any]) -> dict[Any, int]:
    return {item: i for i, item in enumerate(order)}


def _compare(a: dict[Any, float], b: dict[Any, float]) -> tuple[list, int, list[tuple], list[tuple], list[tuple]]:
    """Sort every pair of the items in both `a` and `b` (values: lower is better).

    Returns the common items, the number of pairs in the same order, the pairs
    in opposite order, the pairs `a` ties and the pairs `b` ties. Each pair has
    the item `a` puts higher first.
    """
    items = sorted((x for x in a if x in b), key=lambda x: a[x])
    same, swapped, tied_a, tied_b = 0, [], [], []
    for x, y in combinations(items, 2):
        if a[x] == a[y]:
            tied_a.append((x, y))
        if b[x] == b[y]:
            tied_b.append((x, y))
        if a[x] != a[y] and b[x] != b[y]:
            if b[x] < b[y]:
                same += 1
            else:
                swapped.append((x, y))
    return items, same, swapped, tied_a, tied_b


def kendall_tau_b(values_a: dict[Any, float], values_b: dict[Any, float]) -> float | None:
    """Kendall's tau-b between two {item: value} maps (lower is better), over the items in both.

    Use this where values can tie, such as mean ranks. None if there are fewer
    than two common items or one side ties them all.
    """
    items, same, swapped, tied_a, tied_b = _compare(values_a, values_b)
    pairs = len(items) * (len(items) - 1) // 2
    denominator = math.sqrt((pairs - len(tied_a)) * (pairs - len(tied_b)))
    return (same - len(swapped)) / denominator if denominator else None


def kendall_tau(order_a: Sequence[Any], order_b: Sequence[Any]) -> float | None:
    """Kendall's tau between two orders, best first, over the items in both. None with fewer than two."""
    return kendall_tau_b(_positions(order_a), _positions(order_b))


def swapped_pairs(order_a: Sequence[Any], order_b: Sequence[Any]) -> list[tuple]:
    """The pairs two orders put opposite ways round, each with the item `order_a` puts higher first."""
    return _compare(_positions(order_a), _positions(order_b))[2]


def within_one_adjacent_swap(order_a: Sequence[Any], order_b: Sequence[Any]) -> bool:
    """True if the orders, cut down to the items in both, are identical or differ by one swap of neighbours."""
    in_a, in_b = set(order_a), set(order_b)
    a = [x for x in order_a if x in in_b]
    b = [x for x in order_b if x in in_a]
    different = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return not different or (len(different) == 2 and different[1] == different[0] + 1)


def _agreement(values_a: dict[str, float], values_b: dict[str, float]) -> dict[str, Any]:
    items, _same, swapped, tied_a, tied_b = _compare(values_a, values_b)
    return {
        "n_common": len(items),
        "tau_b": kendall_tau_b(values_a, values_b),
        "swapped_pairs": [list(p) for p in swapped],
        "tied_pairs": [list(p) for p in tied_a + [p for p in tied_b if p not in tied_a]],
        # Breaking ties in the other side's favour leaves exactly the swapped
        # pairs as inversions, and one inversion is always a swap of neighbours.
        "within_one_adjacent_swap": len(swapped) <= 1,
    }


# ---------------------------------------------------------------- intervals from resampling paintings


@cache
def _draws(n: int) -> tuple[tuple[int, ...], ...]:
    """INTERVAL_DRAWS resamples of n paintings, each n positions drawn with replacement. The same for every figure."""
    rng = random.Random(INTERVAL_SEED)
    return tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(INTERVAL_DRAWS))


def _resampled_in_draw_order(by_painting: dict[str, list[float]], paintings: Iterable[str] | None = None) -> list[float | None] | None:
    """The mean of a figure's per-set values under each resample of `paintings`, in the order the resamples are drawn.

    `by_painting` holds, for each painting, the figure's value in each of its sets that has one. `paintings` are
    the paintings drawn from, and may hold more than those: a model with no picture of a painting has no value
    there, and that painting is drawn all the same. Left out, they are the paintings with a value. A resample
    draws as many paintings as there are, with replacement, and the mean is over every value of the paintings
    drawn, a painting drawn twice counting twice. A resample that draws no painting with a value has no mean
    (None). Paintings are taken in name order and the draws come from a fixed seed, so two figures resampled
    over the same paintings meet the same paintings in each resample, and a run always gives the same answer.
    None with fewer than two paintings to draw from, or fewer than two that hold a value.
    """
    if paintings is None:
        paintings = [painting for painting, values in by_painting.items() if values]
    return _ratios_in_draw_order([(sum(by_painting.get(painting, ())), len(by_painting.get(painting, ()))) for painting in sorted(set(paintings))])


def _ratios_in_draw_order(groups: list[tuple[float, int]]) -> list[float | None] | None:
    """A total over a count under each resample of the paintings, in the order the resamples are drawn.

    `groups` holds, for each painting drawn from, in name order, what it adds to the total and what it adds to the
    count: the sum and the number of a figure's set values there (the ratio is then their mean), or a model's wins
    and its comparisons there (the ratio is then its pooled rank score). A painting that adds nothing to the count
    is drawn all the same. A resample whose count is nothing has no ratio (None). None with fewer than two
    paintings to draw from, or fewer than two that add to the count.
    """
    if len(groups) < 2 or sum(1 for _total, count in groups if count) < 2:
        return None
    ratios: list[float | None] = []
    for draw in _draws(len(groups)):
        total, count = 0.0, 0
        for i in draw:
            total += groups[i][0]
            count += groups[i][1]
        ratios.append(total / count if count else None)
    return ratios


def resampled(by_painting: dict[str, list[float]], paintings: Iterable[str] | None = None) -> list[float] | None:
    """The mean of a figure's per-set values under each resample of the paintings, sorted. None with fewer than two.

    `by_painting` holds, for each painting, the figure's value in each of its sets. A resample draws as many
    paintings as there are, with replacement, and the mean is over every set of the paintings drawn, a painting
    drawn twice counting twice. Paintings are taken in name order and the draws come from a fixed seed, so two
    figures over the same paintings are resampled with the same paintings, and a run always gives the same answer.

    `paintings`, when given, are the paintings drawn from: every painting of a list of sets, where the figure may
    have a value at only some of them. A resample that draws none of those has no mean and is left out. Without
    `paintings`, the paintings with a value are the ones drawn from.
    """
    means = _resampled_in_draw_order(by_painting, paintings)
    return None if means is None else sorted(mean for mean in means if mean is not None)


def _percentiles(values: list[float]) -> list[float] | None:
    """[low, high]: the 2.5th and 97.5th percentiles of sorted values, a fortieth of them left out at each end."""
    tail = len(values) // 40
    return [values[tail], values[-1 - tail]] if values else None


def resampling_interval(by_painting: dict[str, list[float]], paintings: Iterable[str] | None = None) -> list[float] | None:
    """[low, high]: the 2.5th and 97.5th percentiles of `resampled`. None with fewer than two paintings."""
    means = resampled(by_painting, paintings)
    return None if means is None else _percentiles(means)


# ---------------------------------------------------------------- Bradley-Terry


def bradley_terry(comparisons: Iterable[tuple[Any, Any]]) -> dict[Any, float]:
    """Bradley-Terry strengths from (winner, loser) pairs, on a log scale.

    P(a beats b) = 1 / (1 + exp(strength[b] - strength[a])). Every item also
    has BT_PRIOR wins and BT_PRIOR losses against a notional opponent, whose
    strength is the zero of the scale; that keeps an unbeaten item finite and
    ties separate groups of items together.

    Fitted by sweeping Zermelo's equations in Newman's form,
    p_i = sum_j w_ij p_j / (p_i + p_j)  /  sum_j w_ji / (p_i + p_j)   (w_ij: times i beat j),
    which settles far sooner than the classic form when strengths are far
    apart. The opponent is swept like any other item, so the overall scale is
    free and does not slow the fit; it is divided out at the end.
    """
    beat: dict[Any, dict[Any, float]] = {_OPPONENT: {}}  # beat[a][b]: times a beat b
    for winner, loser in comparisons:
        if winner == loser:
            continue
        for item in (winner, loser):
            if item not in beat:
                beat[item] = {_OPPONENT: BT_PRIOR}
                beat[_OPPONENT][item] = BT_PRIOR
        beat[winner][loser] = beat[winner].get(loser, 0) + 1
        beat[loser].setdefault(winner, 0)
    if len(beat) == 1:
        return {}
    p = {item: 1.0 for item in beat}
    for _ in range(BT_MAX_SWEEPS):
        change = 0.0
        for i, record in beat.items():
            won = sum(n * p[j] / (p[i] + p[j]) for j, n in record.items())
            lost = sum(beat[j][i] / (p[i] + p[j]) for j in record)
            change = max(change, abs(won / lost - p[i]) / p[i])
            p[i] = won / lost
        if change < BT_TOLERANCE:
            break
    return {item: math.log(p[item] / p[_OPPONENT]) for item in beat if item is not _OPPONENT}


# ---------------------------------------------------------------- reading the run


def current_pictures(layout: RunLayout) -> dict[str, str]:
    """{attempt id: SHA-256 of its picture.txt} for every complete attempt on disk: what a judgement is held against."""
    return {ref.id: ref.picture_sha256() for ref in layout.attempts() if ref.is_complete()}


class Current:
    """What the records of a run are held against, read once for all of them.

    pictures      {attempt id: SHA-256 of its picture.txt} for every complete attempt on disk
    sizes         {(painting, width): how many complete attempts the set holds}
    max_per_call  the config's judge.max_per_call, above which the judge stage judges a set in pairs.
                  None without a config, and then no record is superseded
    unchecked     the paintings whose reference image a record asked after and that could not be read

    The paintings are looked for in `paintings_dir`, else where the config says, else in the default place.
    """

    def __init__(self, layout: RunLayout, cfg: "RunConfig | None" = None, paintings_dir: Path | str | None = None):
        self.pictures = current_pictures(layout)
        self.sizes = Counter(parse_attempt_id(attempt)[:2] for attempt in self.pictures)
        self.max_per_call = cfg.judge.max_per_call if cfg is not None else None
        self.paintings_dir = paintings_dir if paintings_dir is not None else (cfg.paintings_path if cfg is not None else None)
        self.unchecked: set[str] = set()
        self._references: dict[str, str | None] = {}

    def reference(self, painting: str) -> str | None:
        """SHA-256 of the painting's reference image as it is now. None, and the painting noted, if it cannot be read."""
        if painting not in self._references:
            try:
                self._references[painting] = sha256_file(load_painting(painting, self.paintings_dir).reference_path)
            except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):  # no metadata, or none that can be read
                self._references[painting] = None
                self.unchecked.add(painting)
        return self._references[painting]


def _usable(record: dict[str, Any]) -> bool:
    ranked = record.get("ranking_attempts")
    return record.get("status") == "ok" and isinstance(ranked, list) and len(ranked) >= 2


def _without(record: dict[str, Any], aside: set[str]) -> dict[str, Any]:
    """The record with the pictures of the entries kept apart taken out of its ranking; the record itself if it holds none."""
    kept = [attempt for attempt in record["ranking_attempts"] if _model(attempt) not in aside]
    return record if len(kept) == len(record["ranking_attempts"]) else {**record, "ranking_attempts": kept}


def _superseded(record: dict[str, Any], current: Current) -> bool:
    """True for a ranking of a set the judge stage now judges in pairs, and for a pair of a set it now ranks whole."""
    if current.max_per_call is None or record.get("kind") not in ("ranking", "pairwise"):
        return False
    in_pairs = current.sizes[(record.get("painting"), record.get("width"))] > current.max_per_call
    return (record["kind"] == "pairwise") != in_pairs


def _stale_reason(record: dict[str, Any], current: Current) -> str | None:
    """What a record was made on that the run no longer holds: "picture", "reference", or None if nothing.

    "picture": an attempt it says it was shown is no longer complete, or holds another picture.
    "reference": the reference image it says the judge was shown is not the painting's file now.
    A record that does not say is not held against what it does not say.
    """
    shown = record.get("pictures")
    if isinstance(shown, dict) and any(sha is None or current.pictures.get(attempt) != sha for attempt, sha in shown.items()):
        return "picture"
    made_against = record.get("reference")
    if isinstance(made_against, str) and isinstance(record.get("painting"), str):
        now = current.reference(record["painting"])
        if now is not None and now != made_against:  # an image that cannot be read is unknown, not different
            return "reference"
    return None


def is_used(record: dict[str, Any], current: Current) -> bool:
    """The one rule for a judgement record that counts: the scores use it, and the report shows it.

    Its call succeeded and ranked at least two pictures, the judge stage still judges its set by calls
    of its kind, and what it was made on is still in the run. `current` is Current(layout, cfg), read
    once for all the records.
    """
    return _usable(record) and not _superseded(record, current) and _stale_reason(record, current) is None


def _status_unused(record: dict[str, Any], current: Current) -> Any:
    """The status a record that is not used is listed under: its own if the call failed, else why it is left out."""
    if not _usable(record):
        return record.get("status")
    return "superseded" if _superseded(record, current) else "stale"


def is_recorded(human: dict[str, Any]) -> bool:
    """The rule for a person's ranking that counts: it has been recorded, and it orders at least two attempts."""
    ranked = human.get("ranking_attempts")
    return isinstance(ranked, list) and len(ranked) >= 2


def _model(attempt: str) -> str:
    return parse_attempt_id(attempt)[2]


def _call_order(record: dict[str, Any]) -> tuple:
    return (str(record.get("judge")), record.get("repeat") or 0, record.get("_path", ""))


def _group(records: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], tuple]) -> list[tuple[tuple, list[dict[str, Any]]]]:
    """Records grouped by key, groups in key order, each group's calls in judge then repeat order."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return [(k, sorted(groups[k], key=_call_order)) for k in sorted(groups)]


def _settings(attempts: list[tuple[Any, dict[str, Any]]], records: list[dict[str, Any]], cfg: "RunConfig | None", key: str) -> tuple[dict, dict]:
    """{contestant model: value} and {judge: value} of one saved setting. The config only fills in what the records lack."""
    models: dict[str, str] = {}
    judges: dict[str, str] = {}
    for ref, stats in attempts:
        value = (stats.get("model_settings") or {}).get(key)
        if value:
            models.setdefault(ref.model, value)
    for record in records:
        value = (record.get("judge_settings") or {}).get(key)
        if value:
            judges.setdefault(record.get("judge"), value)
    if cfg is not None:
        for found, entries in ((models, cfg.models), (judges, cfg.judges)):
            for entry in entries:
                if getattr(entry, key):
                    found.setdefault(entry.name, getattr(entry, key))
    return models, judges


def _families(attempts: list[tuple[Any, dict[str, Any]]], records: list[dict[str, Any]], cfg: "RunConfig | None") -> tuple[dict, dict]:
    """{contestant model: family} and {judge: family}."""
    return _settings(attempts, records, cfg, "family")


# ---------------------------------------------------------------- rank score and order


def _comparisons(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (winner, loser) a record states: one for a pairwise call, n(n-1)/2 for a ranking of n."""
    return list(combinations(record["ranking_attempts"], 2))


def _score_set(records: list[dict[str, Any]], families: dict[str, str], judge: str | None) -> dict[str, Any]:
    """Score one set: the usable ranking and pairwise records of one judge, or of all judges pooled."""
    mode = "pairwise" if any(r["kind"] == "pairwise" for r in records) else "ranking"
    comparisons = [pair for r in records for pair in _comparisons(r)]
    ranks: dict[str, list[int]] = defaultdict(list)
    if mode == "ranking":
        for r in records:
            for i, attempt in enumerate(r["ranking_attempts"]):
                ranks[attempt].append(i + 1)
    strengths = bradley_terry(comparisons) if mode == "pairwise" else {}

    order = []
    for attempt in sorted({a for r in records for a in r["ranking_attempts"]}):
        _painting, _width, model, n = parse_attempt_id(attempt)
        order.append(
            {
                "attempt": attempt,
                "model": model,
                "n": n,
                "mean_rank": _mean(ranks[attempt]),
                "ranks": ranks[attempt],
                # Rounded so that attempts with the same record tie exactly (+ 0.0 turns -0.0 into 0.0).
                "strength": round(strengths[attempt], 6) + 0.0 if mode == "pairwise" else None,
            }
        )
    order.sort(key=lambda e: (_value(e), e["attempt"]))

    wins: dict[str, int] = defaultdict(int)
    met: dict[str, int] = defaultdict(int)
    for winner, loser in comparisons:
        if _model(winner) != _model(loser):
            wins[_model(winner)] += 1
            met[_model(winner)] += 1
            met[_model(loser)] += 1
    models = []
    for model in sorted({e["model"] for e in order}):
        mine = [e for e in order if e["model"] == model]
        models.append(
            {
                "model": model,
                "family": families.get(model),
                "attempts": len(mine),
                "rank_score": wins[model] / met[model] if met[model] else None,
                "wins": wins[model],
                "comparisons": met[model],
                "mean_rank": _mean(r for e in mine for r in e["ranks"]),
            }
        )
    models.sort(key=_best_score_first("model"))

    compared = {frozenset(pair) for pair in comparisons}
    partial = any(a["model"] != b["model"] and frozenset((a["attempt"], b["attempt"])) not in compared for a, b in combinations(order, 2))

    return {
        "painting": records[0]["painting"],
        "width": records[0]["width"],
        "height": next((r["height"] for r in records if r.get("height")), None),
        "judge": judge,
        "judges": sorted({r["judge"] for r in records}),
        "mode": mode,
        "calls": len(records),
        "common": False,  # decided by _mark_common, once every set it is to be compared with is scored
        "partial_schedule": partial,
        "order": order,
        "models": models,
    }


def _value(entry: dict[str, Any]) -> float:
    """What a set orders its attempts by: lower is better in both modes."""
    return entry["mean_rank"] if entry["strength"] is None else -entry["strength"]


def _values(scored_set: dict[str, Any]) -> dict[str, float]:
    return {e["attempt"]: _value(e) for e in scored_set["order"]}


def _best_score_first(name_key: str) -> Callable[[dict[str, Any]], tuple]:
    return lambda row: (row["rank_score"] is None, -(row["rank_score"] or 0.0), row[name_key])


def _mark_common(sets: list[dict[str, Any]]) -> None:
    """Mark the common sets of one list: those that score every model the list scores anywhere."""
    scored = [{m["model"] for m in s["models"] if m["rank_score"] is not None} for s in sets]
    everyone = set().union(*scored)
    for s, here in zip(sets, scored):
        s["common"] = bool(everyone) and here == everyone


def _scored(scored_set: dict[str, Any]) -> dict[str, float]:
    """{model: rank score} for the models one set scored. A model that met no other model's picture there is not among them."""
    return {m["model"]: m["rank_score"] for m in scored_set["models"] if m["rank_score"] is not None}


def _overall(model: str, sets: list[dict[str, Any]]) -> dict[str, Any]:
    """A model's rank score over every set of one list that scored it, and the same over the list's common sets only."""
    scored = [(s["common"], _scored(s)[model]) for s in sets if model in _scored(s)]
    everywhere = sum(1 for s in sets if _scored(s))
    return {
        "rank_score": _mean(score for _is_common, score in scored),
        "sets_scored": len(scored),
        "sets_all": everywhere,
        "coverage_differs": len(scored) != everywhere,
        "rank_score_common_sets": _mean(score for is_common, score in scored if is_common),
        "sets_common": sum(1 for s in sets if s["common"]),
    }


def _set_scores(sets: list[dict[str, Any]]) -> dict[tuple, dict[str, float]]:
    """{(painting, width): {model: rank score}} for the sets of one list that scored some model: what its overall figures are means of."""
    return {(s["painting"], s["width"]): _scored(s) for s in sets if _scored(s)}


def _by_painting(scores: dict[tuple, dict[str, float]], model: str) -> dict[str, list[float]]:
    """A model's rank score in each set that scored it, grouped by painting, for `resampled`."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for (painting, _width), here in sorted(scores.items()):
        if model in here:
            grouped[painting].append(here[model])
    return grouped


def _paintings(scores: dict[tuple, dict[str, float]]) -> list[str]:
    """The paintings of the sets in `scores`: what an interval over those sets resamples."""
    return sorted({painting for painting, _width in scores})


def _by_width(model: str, combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A model's wins over comparisons at each width, pooled over every set at that width that scored it."""
    rows = []
    for width in sorted({s["width"] for s in combined if _scored(s)}):
        mine = [m for s in combined if s["width"] == width for m in s["models"] if m["model"] == model and m["rank_score"] is not None]
        wins, met = sum(m["wins"] for m in mine), sum(m["comparisons"] for m in mine)
        rows.append({"width": width, "sets": len(mine), "rank_score": wins / met if met else None, "wins": wins, "comparisons": met})
    return rows


def _model_summary(models: list[str], families: dict[str, str], sets: list[dict], combined: list[dict]) -> list[dict[str, Any]]:
    by_judge = {judge: [s for s in sets if s["judge"] == judge] for judge in sorted({s["judge"] for s in sets})}
    pooled = _set_scores(combined)
    rows = []
    for model in models:
        theirs = [{"judge": judge, **_overall(model, own)} for judge, own in by_judge.items()]
        rows.append(
            {
                "model": model,
                "family": families.get(model),
                **_overall(model, combined),
                "by_judge": [j for j in theirs if j["sets_scored"]],
                "rank_score_interval": resampling_interval(_by_painting(pooled, model), _paintings(pooled)),
                "by_width": _by_width(model, combined),
            }
        )
    return sorted(rows, key=_best_score_first("model"))


def _gap(higher: str, lower: str, scores: dict[tuple, dict[str, float]]) -> dict[str, Any]:
    """The difference between two models' figures over one list's sets, each over the sets that scored it, and how firm its sign is.

    Each figure is the mean of that model's rank scores, each set counting once. Every resample draws the same
    paintings for both models; one in which either has no set is left out.
    """
    mine, theirs = _by_painting(scores, higher), _by_painting(scores, lower)
    first, second = (_mean(score for values in grouped.values() for score in values) for grouped in (mine, theirs))
    drawn = [_resampled_in_draw_order(grouped, _paintings(scores)) for grouped in (mine, theirs)]
    return _firmness(first, second, drawn[0], drawn[1])


def _pooled(model: str, sets: list[dict[str, Any]]) -> dict[str, Any]:
    """A model's pooled figure over one list's sets: its wins over its comparisons in the sets that scored it.

    This is the figure a by_width entry holds: `rank_score`, with its `wins`, its `comparisons` and the `sets` that
    scored the model. `drawn` is the same figure under every resample of the list's paintings, in the order the
    resamples are drawn: the wins summed over the comparisons summed, over the sets of the paintings drawn, a painting
    drawn twice counting twice. A resample in which the model has no comparison has no figure (None). `drawn` is None
    where the model has scores at fewer than two paintings.
    """
    paintings = sorted({s["painting"] for s in sets if _scored(s)})
    tally = {painting: [0, 0] for painting in paintings}  # painting -> its wins and its comparisons there
    scored = 0
    for s in sets:
        for m in s["models"]:
            if m["model"] == model and m["rank_score"] is not None:
                tally[s["painting"]][0] += m["wins"]
                tally[s["painting"]][1] += m["comparisons"]
                scored += 1
    wins, met = (sum(here[i] for here in tally.values()) for i in (0, 1))
    drawn = _ratios_in_draw_order([(tally[painting][0], tally[painting][1]) for painting in paintings])
    return {"rank_score": wins / met if met else None, "wins": wins, "comparisons": met, "sets": scored, "drawn": drawn}


def _pooled_gap(higher: dict[str, Any], lower: dict[str, Any]) -> dict[str, Any]:
    """The same as `_gap` for two models' pooled figures over one list's sets (each from `_pooled`).

    Every resample draws the same paintings for both models, and each figure is worked again from the sets of the
    paintings drawn. A resample in which either model has no comparison is left out.
    """
    return _firmness(higher["rank_score"], lower["rank_score"], higher["drawn"], lower["drawn"])


def _difference(first: float | None, second: float | None) -> float | None:
    """One figure minus another, rounded to DIFFERENCE_PLACES so that equal figures differ by exactly zero. None if either is missing.

    A mean of set scores is a sum of floats divided, so two means that are equal as fractions can differ in their
    last bit, and their difference would be about 1e-16 above or below zero by chance. (+ 0.0 turns -0.0 into 0.0.)
    """
    return None if first is None or second is None else round(first - second, DIFFERENCE_PLACES) + 0.0


def _firmness(first: float | None, second: float | None, first_drawn: list[float | None] | None, second_drawn: list[float | None] | None) -> dict[str, Any]:
    """The difference of two figures and how firm its sign is: `difference`, `interval`, `separated` and `share_reversed` of a PAIR.

    `first_drawn` and `second_drawn` hold each figure under every resample, in the order the resamples are drawn, so
    that the same paintings stand behind both; None for a figure with scores at fewer than two paintings, and then
    there is no interval. A resample in which either has no figure is left out. One in which the two are equal has a
    difference of exactly zero, and is not counted as reversed.
    """
    gaps = None
    if first_drawn is not None and second_drawn is not None:
        gaps = sorted(_difference(a, b) for a, b in zip(first_drawn, second_drawn) if a is not None and b is not None)
    interval = _percentiles(gaps) if gaps else None
    return {
        "difference": _difference(first, second),
        "interval": interval,
        "separated": None if interval is None else not interval[0] <= 0 <= interval[1],
        "share_reversed": None if interval is None else sum(1 for gap in gaps if gap < 0) / len(gaps),
    }


def _judges_with_a_figure(model: dict[str, Any]) -> set[str]:
    return {j["judge"] for j in model["by_judge"] if j["rank_score"] is not None}


def _tiers(names: list[str], pairs: list[dict[str, Any]]) -> list[list[str]]:
    """`names`, in order, split wherever two neighbours are separated and nowhere else. `pairs` holds each two neighbours."""
    tiers = [[name] for name in names[:1]]
    for pair in pairs:
        # Only an order that is separated starts a new tier. A pair with no interval (separated is None) has too
        # little to tell by, and a tier of its own for the lower model would present its order as established.
        if pair["separated"] is True:
            tiers.append([pair["lower"]])
        else:
            tiers[-1].append(pair["lower"])
    return tiers


def _standings_by_width(models: list[str], combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The standings at each width with a judged set: the models in order of their pooled figure there, and how firm that order is.

    The figures are the by_width ones, wins over comparisons pooled over the sets there that scored the model. A
    model's interval, and the difference between two neighbours with its interval and the rest, are of those same
    pooled figures (`_pooled`, `_pooled_gap`). Tiers and places follow the rule of the overall order.
    """
    rows = []
    for width in sorted({s["width"] for s in combined if _scored(s)}):
        here = [s for s in combined if s["width"] == width]
        pooled = {model: _pooled(model, here) for model in models}
        ranked = sorted((model for model in models if pooled[model]["rank_score"] is not None), key=lambda model: (-pooled[model]["rank_score"], model))
        pairs = [{"higher": a, "lower": b, **_pooled_gap(pooled[a], pooled[b])} for a, b in zip(ranked, ranked[1:])]
        tiers = _tiers(ranked, pairs)
        places = tier_places({"tiers": tiers})
        entries = []
        for model in ranked:
            mine = pooled[model]
            figures = sorted(figure for figure in mine["drawn"] or [] if figure is not None)
            entries.append(
                {
                    "model": model,
                    "rank_score": mine["rank_score"],
                    "interval": _percentiles(figures),
                    "wins": mine["wins"],
                    "comparisons": mine["comparisons"],
                    "sets_scored": mine["sets"],
                    "place": places[model],
                }
            )
        scored = [s for s in here if _scored(s)]
        rows.append(
            {"width": width, "sets": len(scored), "paintings": len({s["painting"] for s in scored}), "models": entries, "pairs": pairs, "tiers": tiers}
        )
    return rows


def _top_by_width(standings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The two models with the highest rank score at each width, and how firm the order of those two figures is.

    They are the first two of that width's standings, and the difference, its interval and the rest are those of the
    first of its pairs. A width at which fewer than two models have a figure has no entry.
    """
    rows = []
    for at in standings:
        if at["pairs"]:
            first, second = at["models"][:2]
            gap = {key: at["pairs"][0][key] for key in ("difference", "interval", "separated", "share_reversed")}
            rows.append({"width": at["width"], "higher": first["model"], "lower": second["model"], "higher_score": first["rank_score"], "lower_score": second["rank_score"], **gap})
    return rows


def _headline(models: list[dict[str, Any]], sets: list[dict], combined: list[dict], standings: list[dict[str, Any]]) -> dict[str, Any]:
    """How firmly the order of `models` is established: each two neighbours' difference, pooled and by each judge alone."""
    pooled = _set_scores(combined)
    alone = {judge: _set_scores([s for s in sets if s["judge"] == judge]) for judge in sorted({s["judge"] for s in sets})}
    scored = [m for m in models if m["rank_score"] is not None]
    pairs = []
    for a, b in zip(scored, scored[1:]):
        by_judge = []
        for judge in sorted(_judges_with_a_figure(a) & _judges_with_a_figure(b)):
            gap = _gap(a["model"], b["model"], alone[judge])
            by_judge.append({"judge": judge, "difference": gap["difference"], "interval": gap["interval"], "separated": gap["separated"]})
        pair = {"higher": a["model"], "lower": b["model"], **_gap(a["model"], b["model"], pooled), "by_judge": by_judge}
        pair["reversed_by"] = [j["judge"] for j in by_judge if j["difference"] is not None and j["difference"] < 0]
        pairs.append(pair)
    widths = Counter(width for _painting, width in pooled)
    common = Counter(s["width"] for s in combined if s["common"])
    return {
        "draws": INTERVAL_DRAWS,
        "seed": INTERVAL_SEED,
        "level": INTERVAL_LEVEL,
        "paintings": len(_paintings(pooled)),
        "sets": len(pooled),
        "sets_common": sum(common.values()),
        "sets_by_width": [{"width": width, "sets": widths[width], "sets_common": common[width]} for width in sorted(widths)],
        "top_by_width": _top_by_width(standings),
        "pairs": pairs,
        "tiers": _tiers([m["model"] for m in scored], pairs),
    }


# ---------------------------------------------------------------- peak size


def _peak(records: list[dict[str, Any]], judge: str | None, sizes: dict[tuple, dict[int, bool]]) -> dict[str, Any]:
    """`sizes` is {(painting, model): {width: has a complete picture}} over the widths the model was planned or run at."""
    ranks: dict[int, list[int]] = defaultdict(list)
    for r in records:
        for i, attempt in enumerate(r["ranking_attempts"]):
            ranks[parse_attempt_id(attempt)[1]].append(i + 1)
    widths = [{"width": w, "mean_rank": _mean(ranks[w]), "ranked": len(ranks[w])} for w in sorted(ranks)]
    lowest = min(w["mean_rank"] for w in widths)
    best = [w["width"] for w in widths if w["mean_rank"] == lowest]
    painting, model = records[0]["painting"], _peak_model(records[0])
    missing = sorted(w for w, done in sizes.get((painting, model), {}).items() if not done and w not in ranks)
    return {
        "painting": painting,
        "model": model,
        "judge": judge,
        "judges": sorted({r["judge"] for r in records}),
        "calls": len(records),
        "widths": widths,
        "best_widths": best,
        "best_width": best[0] if len(best) == 1 else None,
        "tied": len(best) > 1,
        "widths_not_complete": missing,
        "larger_not_complete": [w for w in missing if w > max(best)],
    }


def _sizes(attempts: list[tuple[Any, dict[str, Any]]], cfg: "RunConfig | None") -> dict[tuple, dict[int, bool]]:
    """{(painting, model): {width: it has a complete picture there}}, over the widths it was planned at or has a record at."""
    sizes: dict[tuple, dict[int, bool]] = defaultdict(dict)
    for painting, width, model, _n in cfg.attempt_jobs() if cfg is not None else ():
        sizes[(painting, model)].setdefault(width, False)
    for ref, stats in attempts:
        here = sizes[(ref.painting, ref.model)]
        here[ref.width] = here.get(ref.width, False) or stats.get("state") == STATE_COMPLETE
    return sizes


def _peak_tally(peaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of `peak_combined` counted by model: how often each width was judged best, ties apart."""
    widths = sorted({w["width"] for p in peaks for w in p["widths"]})
    rows = []
    for model in sorted({p["model"] for p in peaks}):
        mine = [p for p in peaks if p["model"] == model]
        tied = Counter(tuple(p["best_widths"]) for p in mine if p["tied"])
        rows.append(
            {
                "model": model,
                "paintings": len(mine),
                "outright": [{"width": w, "paintings": sum(1 for p in mine if p["best_width"] == w)} for w in widths],
                "tied": [{"widths": list(together), "paintings": n} for together, n in sorted(tied.items())],
                "larger_not_complete": sum(1 for p in mine if p["larger_not_complete"]),
            }
        )
    return rows


# ---------------------------------------------------------------- the rows of a picture


def distinct_rows(picture: Picture) -> int:
    """How many different rows a picture holds, as drawn: two codes for one colour are the same cell."""
    if not picture.palette:
        return len({tuple(row) for row in picture.rows})
    first: dict[tuple, str] = {}
    same = {ord(code): first.setdefault(tuple(colour), code) for code, colour in picture.palette.items()}
    return len({row.translate(same) for row in picture.rows})


def _pictures(attempts: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
    """The rows and the distinct rows of every complete attempt whose picture.txt can be read."""
    rows = []
    for ref, stats in attempts:
        if stats.get("state") != STATE_COMPLETE:
            continue
        try:
            picture = load_picture(ref.picture_txt)
        except (OSError, ValueError):  # no picture file, or one that is not in the format: nothing to count
            continue
        distinct = distinct_rows(picture)
        rows.append({**_named(ref), "rows": picture.height, "distinct_rows": distinct, "share_distinct": distinct / picture.height})
    return rows


def _pictures_by_width(pictures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for picture in pictures:
        grouped[(picture["model"], picture["width"])].append(picture["share_distinct"])
    rows = []
    for (model, width), shares in sorted(grouped.items()):
        under_half = sum(1 for share in shares if share < 0.5)
        rows.append({"model": model, "width": width, "pictures": len(shares), "under_half": under_half, "mean_share_distinct": _mean(shares)})
    return rows


def _peak_model(record: dict[str, Any]) -> str:
    return record.get("model") or _model(record["ranking_attempts"][0])


# ---------------------------------------------------------------- completion and care


def _transcript(ref: AttemptRef) -> dict[str, Any] | None:
    """The attempt's saved transcript; None if there is none, or none that can be read."""
    try:
        data = read_json(ref.transcript_path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def nudges_after_cutoff(ref: AttemptRef, stats: dict[str, Any]) -> int | None:
    """How many of an attempt's nudges followed a reply cut off by the output limit. None if it cannot be known.

    The harness saves the number as "nudges_after_cutoff". For an attempt saved before it did:
    - with no nudge, or no reply cut off, it is 0;
    - otherwise it is counted in transcript.json. A nudge is the harness's message straight after a
      model reply (a reply that called the tool is followed by the tool's result instead), and it
      followed a cut-off when that reply's call stopped for "max_tokens";
    - with no transcript that can be read it is unknown.
    """
    saved = _count(stats.get("nudges_after_cutoff"))
    if saved is not None:
        return saved
    if stats.get("nudges") == 0 or stats.get("truncated_replies") == 0:
        return 0
    transcript = _transcript(ref) or {}
    messages, calls = transcript.get("messages"), transcript.get("calls")
    if not isinstance(messages, list) or not isinstance(calls, list):
        return None
    roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    replies = [i for i, role in enumerate(roles) if role == "assistant"]
    if len(replies) != len(calls):  # not as the harness writes it: one call record for each model reply
        return None
    cut_off = [isinstance(call, dict) and call.get("stop_reason") == STOP_MAX_TOKENS for call in calls]
    return sum(1 for i, cut in zip(replies, cut_off) if cut and roles[i + 1 : i + 2] == ["user"])


def own_nudges(ref: AttemptRef, stats: dict[str, Any]) -> int | None:
    """The attempt's nudges that did not follow a cut-off reply: the care figure. None if either number is unknown."""
    nudges, after = _count(stats.get("nudges")), nudges_after_cutoff(ref, stats)
    if nudges is None or after is None or after > nudges:
        return None
    return nudges - after


def refusal_category(ref: AttemptRef, stats: dict[str, Any]) -> str | None:
    """The provider's category for its block: stop_details.category in the saved response of the attempt's last call.

    None if the provider did not signal the refusal, the response holds no category, or no transcript can be read.
    """
    if stats.get("refusal_detected_by") != "provider":
        return None
    calls = (_transcript(ref) or {}).get("calls")
    last = calls[-1] if isinstance(calls, list) and calls and isinstance(calls[-1], dict) else {}
    raw = last.get("raw") if isinstance(last.get("raw"), dict) else {}
    details = raw.get("stop_details") if isinstance(raw.get("stop_details"), dict) else {}
    category = details.get("category")
    return category if isinstance(category, str) and _CATEGORY.fullmatch(category) else None


def _ending(stats: dict[str, Any], category: str | None) -> dict[str, Any]:
    """How one run of an attempt ended, from its stats record (the ENDING keys)."""
    calls = _count(stats.get("model_calls"))
    return {
        "state": stats.get("state"),
        "stop": stats.get("stop"),
        "refusal_detected_by": stats.get("refusal_detected_by") if stats.get("state") == STATE_REFUSED else None,
        "refusal_category": category,
        "every_reply_cut_off": bool(calls) and stats.get("truncated_replies") == calls,
        "model_calls": calls,
        "truncated_replies": _count(stats.get("truncated_replies")),
        "refusal_text": stats.get("refusal_text"),
        "error": stats.get("error"),
    }


def _completion(stats: list[dict[str, Any]], head: dict[str, Any]) -> dict[str, Any]:
    """One row of the completion tables. `stats` are the attempts' records with "nudges" as own_nudges gives it."""
    typed = [s for s in stats if s.get("state") in (STATE_COMPLETE, STATE_INCOMPLETE)]
    refused = [s.get("refusal_detected_by") for s in stats if s.get("state") == STATE_REFUSED]

    def known(field: str) -> list[float]:
        rows = typed if field in CARE_FIELDS else stats
        return [s[field] for s in rows if isinstance(s.get(field), (int, float)) and not isinstance(s[field], bool)]

    def total(field: str) -> float | None:
        return sum(known(field)) if known(field) else None

    row = {**head, "attempts": len(stats)}
    for state in STATES:
        row[state] = sum(1 for s in stats if s.get("state") == state)
    row["refused_by_provider"] = refused.count("provider")
    row["refused_in_words"] = refused.count("text")
    cut_off = [s for s in stats if (_count(s.get("truncated_replies")) or 0) > 0]
    row["refused_by_provider_after_cut_off"] = sum(1 for s in cut_off if s.get("state") == STATE_REFUSED and s.get("refusal_detected_by") == "provider")
    row["cut_off"] = {state: sum(1 for s in cut_off if s.get("state") == state) for state in STATES}
    row["no_tool_call"] = sum(1 for s in typed if _count(s.get("tool_calls")) == 0)
    row["share_complete"] = row[STATE_COMPLETE] / len(stats) if stats else None
    for field in CARE_FIELDS + ("model_calls",):
        row[f"mean_{field}"] = _mean(known(field))
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        row[field] = total(field)
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        row[f"mean_{field}"] = _mean(known(field))
    row["cost_usd"] = total("cost_usd")
    row["mean_wall_seconds"] = _mean(known("wall_seconds"))
    row["known"] = {field: len(known(field)) for field in CARE_FIELDS + USAGE_FIELDS}
    return row


def _completion_tables(attempts: list[tuple[Any, dict[str, Any]]], families: dict[str, str]) -> dict[str, Any]:
    by_model: dict[str, list] = defaultdict(list)
    by_cell: dict[tuple, list] = defaultdict(list)
    by_model_width: dict[tuple, list] = defaultdict(list)
    by_width: dict[int, list] = defaultdict(list)
    # The care figure for nudges leaves out those that followed a cut-off reply: they say nothing about care.
    attempts = [(ref, {**stats, "nudges": own_nudges(ref, stats)}) for ref, stats in attempts]
    for ref, stats in attempts:
        by_model[ref.model].append(stats)
        by_cell[(ref.model, ref.painting, ref.width)].append(stats)
        by_model_width[(ref.model, ref.width)].append(stats)
        by_width[ref.width].append(stats)
    return {
        "total": _completion([stats for _ref, stats in attempts], {}),
        "by_model": [_completion(by_model[m], {"model": m, "family": families.get(m)}) for m in sorted(by_model)],
        "by_cell": [
            _completion(by_cell[key], {"model": key[0], "family": families.get(key[0]), "painting": key[1], "width": key[2]})
            for key in sorted(by_cell)
        ],
        "by_model_width": [
            _completion(by_model_width[key], {"model": key[0], "family": families.get(key[0]), "width": key[1]}) for key in sorted(by_model_width)
        ],
        "by_width": [_completion(by_width[width], {"width": width}) for width in sorted(by_width)],
    }


def _planned(cfg: "RunConfig | None", layout: RunLayout, attempts: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any] | None:
    """The planned grid of the config against the attempts with a record. None without a config: the plan is not known.

    A planned attempt whose stats.json is on disk and cannot be read is neither run nor never started here: the
    listing has already named it in a warning.
    """
    if cfg is None:
        return None
    on_disk = {ref.id for ref, _stats in attempts}
    jobs = sorted(set(cfg.attempt_jobs()))
    missing = [job for job in jobs if attempt_id(*job) not in on_disk and not layout.attempt(*job).stats_path.exists()]

    def tally(key: Callable[[tuple], tuple], names: tuple[str, ...]) -> list[dict[str, Any]]:
        planned, unstarted = Counter(key(job) for job in jobs), Counter(key(job) for job in missing)
        run = Counter(key(job) for job in jobs if attempt_id(*job) in on_disk)
        return [{**dict(zip(names, k)), "planned": planned[k], "run": run[k], "not_started": unstarted[k]} for k in sorted(planned)]

    return {
        "attempts": len(jobs),
        "run": sum(1 for job in jobs if attempt_id(*job) in on_disk),
        "not_started": [
            {"attempt": attempt_id(painting, width, model, n), "painting": painting, "width": width, "model": model, "n": n}
            for painting, width, model, n in missing
        ],
        "unplanned": len(on_disk - {attempt_id(*job) for job in jobs}),
        "by_model": tally(lambda job: (job[2],), ("model",)),
        "by_model_width": tally(lambda job: (job[2], job[1]), ("model", "width")),
        "by_width": tally(lambda job: (job[1],), ("width",)),
    }


def _named(ref: AttemptRef) -> dict[str, Any]:
    return {"attempt": ref.id, "painting": ref.painting, "width": ref.width, "model": ref.model, "n": ref.n}


def _not_complete(attempts: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {**_named(ref), **_ending(stats, refusal_category(ref, stats))}
        for ref, stats in attempts
        if stats.get("state") != STATE_COMPLETE
    ]


def _retried(attempts: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
    """The attempts that replaced an earlier run of themselves, each with how those runs ended, oldest first."""
    rows = []
    for ref, stats in attempts:
        earlier = stats.get("earlier")
        earlier = [e for e in earlier if isinstance(e, dict)] if isinstance(earlier, list) else []
        if earlier:
            # An earlier run's transcript was replaced with it, so its category can no longer be read.
            runs = [{**_ending(e, None), "finished_at": e.get("finished_at")} for e in earlier]
            rows.append({**_named(ref), "state": stats.get("state"), "stop": stats.get("stop"), "earlier": runs})
    return rows


# ---------------------------------------------------------------- judge versus person, judge versus judge


def _repeat_agreement(person_order: list[str], record: dict[str, Any]) -> dict[str, Any]:
    judge_order = record["ranking_attempts"]
    swapped = swapped_pairs(person_order, judge_order)
    return {
        "repeat": record.get("repeat"),
        "judge_order": list(judge_order),
        "n_common": len(set(person_order) & set(judge_order)),
        "tau": kendall_tau(person_order, judge_order),
        "swapped_pairs": [list(p) for p in swapped],
        "identical": not swapped,
        "within_one_adjacent_swap": within_one_adjacent_swap(person_order, judge_order),
    }


def _judge_vs_person(people: list[dict[str, Any]], sets: list[dict], calls: dict[tuple, list[dict]]) -> list[dict[str, Any]]:
    rows = []
    for human in people:
        person_order = list(human["ranking_attempts"])
        for s in sets:
            if (s["painting"], s["width"]) != (human["painting"], human["width"]):
                continue
            combined = _agreement(_positions(person_order), _values(s))
            if combined["n_common"] < 2:
                continue
            rankings = [r for r in calls[(s["painting"], s["width"], s["judge"])] if r["kind"] == "ranking"]
            repeats = [r for r in (_repeat_agreement(person_order, call) for call in rankings) if r["n_common"] >= 2]
            rows.append(
                {
                    "painting": s["painting"],
                    "width": s["width"],
                    "person": human["person"],
                    "judge": s["judge"],
                    "mode": s["mode"],
                    "person_order": person_order,
                    "repeats": repeats,
                    "repeats_compared": len(repeats),
                    "repeats_identical": sum(r["identical"] for r in repeats),
                    "repeats_within_one_adjacent_swap": sum(r["within_one_adjacent_swap"] for r in repeats),
                    "combined": combined,
                }
            )
    return rows


def _judge_vs_judge(sets: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for a, b in combinations(sets, 2):  # sets are sorted, so judge_a sorts before judge_b
        if (a["painting"], a["width"]) == (b["painting"], b["width"]):
            rows.append({"painting": a["painting"], "width": a["width"], "judge_a": a["judge"], "judge_b": b["judge"], **_agreement(_values(a), _values(b))})
    return rows


def _own_family(sets: list[dict], judge_families: dict[str, str], families: dict[str, str]) -> list[dict[str, Any]]:
    cells: dict[tuple, dict[str, dict[str, float]]] = defaultdict(dict)  # (painting, width) -> judge -> model -> score
    for s in sets:
        cells[(s["painting"], s["width"])][s["judge"]] = {m["model"]: m["rank_score"] for m in s["models"] if m["rank_score"] is not None}
    rows = []
    judges = sorted({s["judge"] for s in sets})
    for judge in judges:
        family = judge_families.get(judge)
        # A judge of the same family may share the favour looked for, so it is no baseline for it.
        same_family = [j for j in judges if j != judge and family and judge_families.get(j) == family]
        mine_scores, their_scores, models, others = [], [], set(), set()
        for cell in cells.values():
            mine = cell.get(judge, {})
            own = [m for m in mine if family and families.get(m) == family]
            if len(own) == len(mine):
                continue  # no outsider to rank them against: the mean says nothing about favour
            for model in own:
                theirs = {j: scores[model] for j, scores in cell.items() if j != judge and j not in same_family and model in scores}
                if theirs:
                    mine_scores.append(mine[model])
                    their_scores.append(_mean(theirs.values()))
                    models.add(model)
                    others.update(theirs)
        own_mean, others_mean = _mean(mine_scores), _mean(their_scores)
        rows.append(
            {
                "judge": judge,
                "family": family,
                "own_models": sorted(models),
                "other_judges": sorted(others),
                "same_family_judges": same_family,
                "scores_compared": len(mine_scores),
                "own_mean": own_mean,
                "others_mean": others_mean,
                "difference": own_mean - others_mean if mine_scores else None,
            }
        )
    return rows


def _own_model(sets: list[dict], judge_ids: dict[str, str], model_ids: dict[str, str]) -> list[dict[str, Any]]:
    """Each judge that is also a contestant, on that contestant: its score from this judge against the other judges'.

    The comparison is paired set by set, over every set in which this judge and at least one other judge scored the
    contestant. A set in which some other model has no picture is such a set like any other. The judges are matched on
    the painting and the width only: they need not have ranked the contestant against the same pictures there.
    """
    cells: dict[tuple, dict[str, dict[str, float]]] = defaultdict(dict)  # (painting, width) -> judge -> model -> score
    for s in sets:
        cells[(s["painting"], s["width"])][s["judge"]] = _scored(s)
    rows = []
    for judge in sorted({s["judge"] for s in sets}):
        for contestant in sorted(m for m, model_id in model_ids.items() if judge_ids.get(judge) and model_id == judge_ids[judge]):
            own: dict[tuple, float] = {}
            theirs: dict[tuple, float] = {}
            others: set[str] = set()
            for key, cell in cells.items():
                rest = {j: scores[contestant] for j, scores in cell.items() if j != judge and contestant in scores}
                if contestant in cell.get(judge, {}) and rest:
                    own[key], theirs[key] = cell[judge][contestant], _mean(rest.values())
                    others.update(rest)
            # Rounded as every difference is: the mean of 2/5 and 4/5 is not exactly 3/5 as floats, and a set in which
            # the judge and the others agree must count as equal, not as lower by a hair.
            gaps = {key: _difference(own[key], theirs[key]) for key in own}
            grouped: dict[str, list[float]] = defaultdict(list)
            for (painting, _width), gap in sorted(gaps.items()):
                grouped[painting].append(gap)
            rows.append(
                {
                    "judge": judge,
                    "contestant": contestant,
                    "model_id": judge_ids[judge],
                    "other_judges": sorted(others),
                    "sets": len(gaps),
                    "own_score": _mean(own.values()),
                    "others_score": _mean(theirs.values()),
                    "difference": _mean(gaps.values()),
                    "interval": resampling_interval(grouped),
                    "sets_higher": sum(1 for gap in gaps.values() if gap > 0),
                    "sets_lower": sum(1 for gap in gaps.values() if gap < 0),
                    "sets_equal": sum(1 for gap in gaps.values() if gap == 0),
                }
            )
    return rows


def _judges(used: list[dict[str, Any]], judge_ids: dict[str, str], judge_families: dict[str, str], model_ids: dict[str, str]) -> list[dict[str, Any]]:
    """What each judge is, and what its records in use say of how its calls went."""
    rows = []
    for (judge,), records in _group(used, lambda r: (str(r.get("judge")),)):
        retries = [_count((r.get("usage") or {}).get("api_retries")) if isinstance(r.get("usage"), dict) else None for r in records]
        retries = [n for n in retries if n is not None]
        ordered = [r for r in records if isinstance(r.get("order_shown"), list) and r["order_shown"] and isinstance(r.get("ranking"), list) and r["ranking"]]
        rows.append(
            {
                "judge": judge,
                "model": judge_ids.get(judge),
                "family": judge_families.get(judge),
                "contestants": sorted(m for m, model_id in model_ids.items() if judge_ids.get(judge) and model_id == judge_ids[judge]),
                "calls": len(records),
                "calls_reporting_api_retries": len(retries),
                "calls_with_api_retries": sum(1 for n in retries if n > 0),
                "api_retries": sum(retries),
                "calls_without_reference": sum(1 for r in records if "reference" not in r),
                "calls_with_order": len(ordered),
                "first_shown_best": sum(1 for r in ordered if r["ranking"][0] == r["order_shown"][0]),
                "first_shown_expected": sum(1 / len(r["ranking"]) for r in ordered),
            }
        )
    return rows


def _judge_agreement(calls: dict[tuple, list[dict]], judge_vs_judge: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement at three levels: the judges' averaged orders, single calls of different judges, one judge's own repeats."""
    rankings: dict[tuple, dict[str, list[list[str]]]] = defaultdict(dict)  # (painting, width) -> judge -> its rankings
    for (painting, width, judge), records in calls.items():
        orders = [r["ranking_attempts"] for r in records if r["kind"] == "ranking"]
        if orders:
            rankings[(painting, width)][judge] = orders
    between: list[float] = []
    between_sets = 0
    within: dict[str, list[float]] = defaultdict(list)
    within_sets: Counter = Counter()
    for _key, by_judge in sorted(rankings.items()):
        found = [tau for a, b in combinations(sorted(by_judge), 2) for x in by_judge[a] for y in by_judge[b] if (tau := kendall_tau(x, y)) is not None]
        between += found
        between_sets += bool(found)
        for judge, orders in by_judge.items():
            found = [tau for x, y in combinations(orders, 2) if (tau := kendall_tau(x, y)) is not None]
            within[judge] += found
            within_sets[judge] += bool(found)
    taus = [row["tau_b"] for row in judge_vs_judge if row["tau_b"] is not None]
    return {
        "between_orders": {"tau_b": _mean(taus), "sets": len(taus), "lowest": min(taus) if taus else None},
        "between_calls": {"tau": _mean(between), "pairs": len(between), "sets": between_sets},
        "within": [
            {"judge": judge, "tau": _mean(within[judge]), "pairs": len(within[judge]), "sets": within_sets[judge]}
            for judge in sorted(within)
            if within[judge]
        ],
    }


# ---------------------------------------------------------------- the whole run


def _grids(attempts: list[tuple[Any, dict[str, Any]]], sets: list[dict], people: list[dict]) -> list[dict[str, Any]]:
    grids: dict[tuple, dict[str, Any]] = {}

    def grid(painting: str, width: int) -> dict[str, Any]:
        empty = {"painting": painting, "width": width, "height": None, "complete": [], "not_judged": [], "judges": [], "people": []}
        return grids.setdefault((painting, width), empty)

    for ref, stats in attempts:
        g = grid(ref.painting, ref.width)
        g["height"] = g["height"] or stats.get("height")
        if stats.get("state") == STATE_COMPLETE:
            g["complete"].append(ref.id)
    for s in sets:
        g = grid(s["painting"], s["width"])
        g["height"] = g["height"] or s["height"]
        g["judges"].append(s["judge"])
    for human in people:
        g = grid(human["painting"], human["width"])
        g["height"] = g["height"] or human.get("height")
        g["people"].append(human["person"])
    ranked = {e["attempt"] for s in sets for e in s["order"]}
    for g in grids.values():
        g["not_judged"] = [attempt for attempt in g["complete"] if attempt not in ranked]
    return [grids[key] for key in sorted(grids)]


def _failed(record: dict[str, Any], status: Any, current: Current, root: Path) -> dict[str, Any]:
    return {
        "kind": record.get("kind"),
        "painting": record.get("painting"),
        "width": record.get("width"),
        "judge": record.get("judge"),
        "model": record.get("model"),
        "repeat": record.get("repeat"),
        "status": status,
        "stale_reason": _stale_reason(record, current) if status == "stale" else None,
        "parse_errors": list(record.get("parse_errors") or []),
        "error": record.get("error"),
        "path": os.path.relpath(record["_path"], root),
    }


def compute_scores(layout: RunLayout, cfg: "RunConfig | None" = None, paintings_dir: Path | str | None = None) -> dict[str, Any]:
    """Everything the report shows, from the files of one run. Reads only.

    `cfg` is optional and is used for four things: a model's or judge's
    family and model id, where the attempt's or judgement's own saved settings
    do not carry one; judge.max_per_call, without which no record is
    superseded; the planned grid, without which `planned` is null; and where
    the paintings are. The paintings are needed only to hold a judgement
    against the reference image it was made with. They are looked for in
    `paintings_dir`, else where the config says, else in the default place.
    """
    attempts = [(ref, ref.stats() or {}) for ref in layout.attempts()]
    records = layout.judgements()
    current = Current(layout, cfg, paintings_dir)
    used = [r for r in records if is_used(r, current)]
    unused = [(r, _status_unused(r, current)) for r in records if r not in used]
    stale = sum(1 for _r, status in unused if status == "stale")
    superseded = sum(1 for _r, status in unused if status == "superseded")
    humans = layout.human_rankings()
    recorded = [h for h in humans if is_recorded(h)]
    # A person's ranking of pictures that have since changed says nothing about the pictures now on disk.
    people = sorted((h for h in recorded if _stale_reason(h, current) is None), key=lambda h: (h["painting"], h["width"], h["person"]))
    families, judge_families = _families(attempts, records, cfg)
    model_ids, judge_ids = _settings(attempts, records, cfg, "model")

    # An entry the config keeps apart is in no comparison: its pictures are taken out of every ranking before the
    # ranking is scored, which leaves the order of the others as the judge gave it.
    aside = set(cfg.aside) if cfg is not None else set()
    same_size = [w for w in (_without(r, aside) for r in used if r["kind"] != "peak") if len(w["ranking_attempts"]) >= 2]
    calls = dict(_group(same_size, lambda r: (r["painting"], r["width"], r["judge"])))
    sets = [_score_set(recs, families, key[2]) for key, recs in calls.items()]
    combined = [_score_set(recs, families, None) for _key, recs in _group(same_size, lambda r: (r["painting"], r["width"]))]
    _mark_common(combined)
    for judge in {s["judge"] for s in sets}:
        _mark_common([s for s in sets if s["judge"] == judge])
    peaks = [r for r in used if r["kind"] == "peak"]
    models = sorted({ref.model for ref, _stats in attempts} | {m["model"] for s in combined for m in s["models"]})
    summary = _model_summary(models, families, sets, combined)
    sizes = _sizes(attempts, cfg)
    peak_combined = [_peak(recs, None, sizes) for _key, recs in _group(peaks, lambda r: (r["painting"], _peak_model(r)))]
    pictures = _pictures(attempts)
    judge_vs_judge = _judge_vs_judge(sets)
    standings = _standings_by_width(models, combined)

    return {
        "scores_version": SCORES_VERSION,
        "harness_version": __version__,
        "run_id": layout.root.name,
        "counts": {
            "attempts": len(attempts),
            "judgements": len(used),
            "judgements_failed": len(unused) - stale - superseded,
            "judgements_stale": stale,
            "judgements_superseded": superseded,
            "human_rankings": len(people),
            "human_rankings_pending": len(humans) - len(recorded),
            "human_rankings_stale": len(recorded) - len(people),
        },
        "aside": sorted(aside),  # the entries kept apart: no rank score, and in nobody else's
        "planned": _planned(cfg, layout, attempts),
        "grids": _grids(attempts, sets, people),
        "sets": sets,
        "combined": combined,
        "models": summary,
        "headline": _headline(summary, sets, combined, standings),
        "default_width": standings[-1]["width"] if standings else None,  # the largest: the widths are in ascending order
        "standings_by_width": standings,
        "peak": [_peak(recs, key[2], sizes) for key, recs in _group(peaks, lambda r: (r["painting"], _peak_model(r), r["judge"]))],
        "peak_combined": peak_combined,
        "peak_tally": _peak_tally(peak_combined),
        "pictures": pictures,
        "pictures_by_width": _pictures_by_width(pictures),
        "completion": _completion_tables(attempts, families),
        "not_complete": _not_complete(attempts),
        "retried": _retried(attempts),
        "judge_vs_person": _judge_vs_person(people, sets, calls),
        "judge_vs_judge": judge_vs_judge,
        "own_family": _own_family(sets, judge_families, families),
        "own_model": _own_model(sets, judge_ids, model_ids),
        "judges": _judges(used, judge_ids, judge_families, model_ids),
        "judge_agreement": _judge_agreement(calls, judge_vs_judge),
        "failed_judgements": [_failed(r, status, current, layout.root) for r, status in unused],
        "reference_unchecked": sorted(current.unchecked),  # every record has been held against its reference by now
    }


def write_scores(layout: RunLayout, scores: dict[str, Any]) -> Path:
    write_json(layout.scores_path, scores)
    return layout.scores_path


# ---------------------------------------------------------------- the terminal summary


def _num(value: float | None, places: int = 2) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Aligned columns: the first left, the rest right."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    lines = []
    for row in [header] + rows:
        cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        lines.append("  " + "  ".join(cells).rstrip())
    return lines


def _short(attempt: str) -> str:
    """'<model>/<n>': enough to name a picture inside one painting and width."""
    return "/".join(attempt.split("/")[2:])


def _pairs_text(pairs: list[list[str]], limit: int = 4) -> str:
    if not pairs:
        return "none"
    shown = ", ".join(f"{_short(a)} <> {_short(b)}" for a, b in pairs[:limit])
    return shown if len(pairs) <= limit else f"{shown} and {len(pairs) - limit} more"


def _names_text(attempts: list[str], limit: int = 6) -> str:
    shown = ", ".join(_short(a) for a in attempts[:limit])
    return shown if len(attempts) <= limit else f"{shown} and {len(attempts) - limit} more"


def _starred(entry: dict[str, Any]) -> bool:
    """True for an overall figure (a MODEL entry or one of its by_judge) of a model that some set of its list did not score."""
    return entry["rank_score"] is not None and entry["coverage_differs"]


def _overall_text(entry: dict[str, Any] | None, star: str) -> str:
    """The figure, with `star` after it if it is starred and as much space if not, so a column's figures stay in line."""
    return ("-" if entry is None else _num(entry["rank_score"])) + (star if entry and _starred(entry) else " " * len(star))


def _signed(value: float | None, places: int = 2) -> str:
    """A difference with its sign; one that rounds to nothing is written without one, so that it does not read as measured."""
    if value is None:
        return "-"
    text = f"{value:+.{places}f}"
    return text[1:] if float(text) == 0 else text


def _range_text(interval: list[float] | None, signed: bool = False, places: int = 2) -> str:
    """'0.81 to 0.90' for an interval; '-' where there is none."""
    if interval is None:
        return "-"
    return " to ".join(_signed(end, places) if signed else _num(end, places) for end in interval)


def _widths_text(widths: list[int]) -> str:
    return " and ".join(str(w) for w in widths)


def percent_text(share: float | None, unknown: str = "-") -> str:
    """A share as a whole percentage. One that is not exactly 0 or 1 but would round to it gets a decimal: '99.5%'."""
    if share is None:
        return unknown
    text = f"{100 * share:.0f}%"
    if text not in ("0%", "100%") or share in (0, 1):
        return text
    text = f"{100 * share:.1f}%"
    return {"0.0%": "under 0.1%", "100.0%": "over 99.9%"}.get(text, text)


def tier_places(headline: dict[str, Any]) -> dict[str, str]:
    """{model: its place}: '1', or '3-4' for each of the neighbours whose order is not separated, or has no interval to tell by.

    From the tiers of the overall order (`headline`) or of one width's standings (an entry of `standings_by_width`).
    """
    places, first = {}, 1
    for tier in headline["tiers"]:
        last = first + len(tier) - 1
        for model in tier:
            places[model] = str(first) if len(tier) == 1 else f"{first}-{last}"
        first = last + 1
    return places


def means_repeat_standings(scores: dict[str, Any]) -> bool:
    """True where a table of the means over every set would only repeat the standings: there is nothing more in it to read.

    That needs a run of one width, and the same models, figures, intervals and neighbours' differences in both. Having
    the same sets is not enough: a mean counts each set once and a pooled figure each comparison once, so the two
    differ wherever the sets of a model hold different numbers of comparisons (a repeat that failed, say).
    """
    standings = scores["standings_by_width"]
    if len(standings) != 1:
        return False

    def near(value: Any) -> Any:
        return [near(end) for end in value] if isinstance(value, list) else value if value is None else round(value, 9)

    means = [(m["model"], near(m["rank_score"]), near(m["rank_score_interval"])) for m in scores["models"] if m["rank_score"] is not None]
    pooled = [(m["model"], near(m["rank_score"]), near(m["interval"])) for m in standings[0]["models"]]
    gaps = [[(pair["higher"], pair["lower"], near(pair["difference"]), near(pair["interval"])) for pair in pairs]
            for pairs in (scores["headline"]["pairs"], standings[0]["pairs"])]
    return means == pooled and gaps[0] == gaps[1]


def _cut_off_before_block(ending: dict[str, Any]) -> bool:
    """True for a provider's block that came after at least one earlier reply had been cut off by the output limit."""
    blocked = ending.get("state") == STATE_REFUSED and ending.get("refusal_detected_by") == "provider"
    return blocked and bool(ending.get("truncated_replies")) and bool(ending.get("model_calls"))


def _ending_text(ending: dict[str, Any]) -> str:
    """'<state> (<why>)' for one run of an attempt (an ENDING). A provider's block is never worded as the model declining."""
    why = ending["stop"]
    if ending["state"] == STATE_REFUSED and ending["refusal_detected_by"] == "provider":
        why = "blocked by the provider" + (f": {ending['refusal_category']}" if ending["refusal_category"] else "")
        if _cut_off_before_block(ending):
            why += f", after the output limit cut off {ending['truncated_replies']} of its {ending['model_calls']} replies"
    elif ending["state"] == STATE_REFUSED and ending["refusal_detected_by"] == "text":
        why = "the model declined in words"
    elif ending["stop"] == "nudges_exhausted" and ending["every_reply_cut_off"]:
        why = "nudges_exhausted: every reply was cut off by the output limit"
    return f"{ending['state']} ({why})"


def _by_directory(failed: list[dict[str, Any]]) -> list[str]:
    """One re-typed picture makes every call that showed it stale, so such records are counted by directory, not listed."""
    return [f"  {where}: {n}" for where, n in sorted(Counter(os.path.dirname(f["path"]) for f in failed).items())]


def _interval_text(paintings: int | None, draws: int) -> str:
    """What an interval is. `paintings` is how many are resampled, None where the line speaks for tables that resample different numbers."""
    which = "the paintings of each table" if paintings is None else f"the {paintings} paintings"
    return (
        f"  95% interval: from resampling {which} ({draws} draws). With one attempt per model "
        "and set it does not include attempt-to-attempt variation, so it is if anything too narrow."
    )


# The neighbours an order leaves open: what each kind is called, and why its two models are not placed apart.
_OPEN = {
    False: ("Not separated", "the interval of the difference includes zero, so the order could go either way"),
    None: ("Too little to tell", "the difference has no interval, so the order is neither separated nor shown to be level"),
}


def _open_pairs_text(pairs: list[dict[str, Any]], lacking: set[str], name_unknown: bool, said: set[str], places: bool = True) -> list[str]:
    """The neighbours of one order that it does not separate, and those it has too little to tell apart, a line each.

    `lacking` are the models with no interval. A pair with no interval is listed only with `name_unknown`: where no
    difference of the order can have one, a single line says so of all of them. `said` holds what an earlier table
    has explained already, and each kind is explained the first time it comes up. `places` is whether the order
    gives places, which such neighbours then share.
    """
    lines = []
    for separated in (False, None):
        these = [pair for pair in pairs if pair["separated"] is separated] if separated is False or name_unknown else []
        if not these:
            continue
        name, why = _OPEN[separated]
        lines.append(f"  {name}:" if name in said else f"  {name} ({'a shared place: ' if places else ''}{why}):")
        said.add(name)
        for pair in these:
            line = f"    {pair['higher']} and {pair['lower']}: difference {_signed(pair['difference'])}, "
            if separated is False:
                line += f"interval {_range_text(pair['interval'], True)}, reversed in {percent_text(pair['share_reversed'])} of resamples"
            else:
                few = [name for name in (pair["higher"], pair["lower"]) if name in lacking]
                line += f"no interval ({' and '.join(few)} scored on fewer than two paintings)" if few else "no interval (too few resamples hold both)"
            lines.append(line)
    return lines


def _standings_text(
    at: dict[str, Any], headline: dict[str, Any], lead: bool, several: bool, said: set[str], drawn: int | None = None, fewer: bool = False
) -> list[str]:
    """The standings at one width (an entry of standings_by_width) as a table, with the neighbours its order leaves open.

    `lead` is the table a summary opens with: what holds of every such table is said under it, and nowhere else.
    `several` is whether the run has other widths. `said` is as in `_open_pairs_text`. `drawn` is how many paintings
    every table of the summary resamples, None where they do not all resample as many. `fewer` is whether, at some
    width, a model's figure rests on fewer sets than the width has.
    """
    lines = [f"Standings at {at['width']} wide ({at['sets']} {'set' if at['sets'] == 1 else 'sets'})"]
    rows = [[m["model"], m["place"], _num(m["rank_score"]), _range_text(m["interval"]), str(m["sets_scored"])] for m in at["models"]]
    lines += _table(["model", "place", "rank score", "95% interval", "sets"], rows)
    intervals = any(m["interval"] for m in at["models"])
    if not intervals and "no interval" in said:
        lines.append("  No interval at this width either" + (", so its neighbours share a place." if at["pairs"] else "."))
    elif not intervals and at["paintings"] >= 2:
        lines.append(
            "  No interval: it comes from resampling paintings, and needs a model to have scores at two or more. This width has "
            f"{at['paintings']} paintings, but no model was scored on more than one of them."
        )
    elif not intervals:
        lines.append(
            "  No interval: it comes from resampling paintings, and needs at least two."
            + (" Without one there is too little to tell the order of any two neighbours, so they share a place." if at["pairs"] else "")
        )
    if not intervals:
        said.add("no interval")
    # A difference with no interval, at a width with two paintings or more: one of the two has scores at fewer than two of them.
    # (At a width with one painting no difference has an interval, and the line above says so of all of them.)
    lines += _open_pairs_text(at["pairs"], {m["model"] for m in at["models"] if not m["interval"]}, intervals or at["paintings"] >= 2, said)
    if lead:
        lines.append(
            "  Each figure is the model's wins over its comparisons at that width, pooled over judges and over the sets there that "
            "scored it (the sets column). A set in which a model has no complete picture is no loss to it."
            + (" A figure on fewer sets is not over the same paintings as the others." if fewer else "")
        )
        lines.append(
            "  A rank score is a share of comparisons won inside this run, against this run's own pictures: figures from "
            "different runs cannot be set against each other."
        )
        if intervals:
            lines.append(_interval_text(drawn, headline["draws"]))
            said.add("interval")
        if several:
            lines.append(
                "  Each width has its own standings, the largest first: a larger grid asks more of a model, and a standing can "
                "fall as the grid grows."
            )
    return lines


def format_scores(scores: dict[str, Any]) -> str:
    """A compact plain-text summary for the terminal. Holds no text written by a model."""
    counts, total = scores["counts"], scores["completion"]["total"]
    superseded = f", superseded {counts['judgements_superseded']}" if counts["judgements_superseded"] else ""
    lines = [
        f"Run {scores['run_id']}: attempts {counts['attempts']} ({total['complete']} complete), "
        f"judgements used {counts['judgements']}, failed {counts['judgements_failed']}, stale {counts['judgements_stale']}{superseded}, "
        f"human rankings {counts['human_rankings']}."
    ]
    planned = scores["planned"]
    if planned and planned["not_started"]:
        whose = ", ".join(f"{m['model']} {m['not_started']}" for m in planned["by_model"] if m["not_started"])
        lines.append(
            f"{planned['run']} of {planned['attempts']} planned attempts were run; {len(planned['not_started'])} "
            f"{'was' if len(planned['not_started']) == 1 else 'were'} never started ({whose}). No reason is recorded for them."
        )

    lines += ["", "Rank score: share of comparisons won against other models' pictures (1 always best, 0 always worst)"]
    judges = sorted({s["judge"] for s in scores["sets"]})
    models = scores["models"]
    headline = scores["headline"]
    # The largest width leads, and the others follow it, largest first. What holds of every table is said under the first.
    standings = sorted(scores["standings_by_width"], key=lambda at: -at["width"])
    said: set[str] = set()
    # The one line on the interval names a number of paintings only where every table resamples as many.
    resampled_by = {at["paintings"] for at in standings} | {headline["paintings"]}
    drawn = resampled_by.pop() if len(resampled_by) == 1 else None
    fewer = any(m["sets_scored"] < at["sets"] for at in standings for m in at["models"])
    for at in standings:
        lines += ([] if at is standings[0] else [""]) + _standings_text(at, headline, at is standings[0], len(standings) > 1, said, drawn, fewer)
    if not judges:
        lines.append("  no judgements yet")

    widths = headline["sets_by_width"]
    if len(widths) > 1:
        # The same figures side by side, in the order of the leading table: how a model's standing moves with the size.
        leading = [m["model"] for m in standings[0]["models"]] if standings else []
        across = sorted(models, key=lambda m: leading.index(m["model"]) if m["model"] in leading else len(leading))
        lines += ["", "Rank score by width (the figures of the standings side by side)"]
        header = ["model"] + [f"{w['width']} ({w['sets']} {'set' if w['sets'] == 1 else 'sets'})" for w in widths]
        at_width = {(m["model"], w["width"]): w for m in models for w in m["by_width"]}
        columns = []
        for w in widths:
            mine = [at_width.get((m["model"], w["width"])) for m in across]
            # A model that some set at this width did not score has its own number of sets after its figure.
            fewer = [f" ({here['sets']})" if here and here["rank_score"] is not None and here["sets"] != w["sets"] else "" for here in mine]
            room = max(len(count) for count in fewer)  # as much space after the other figures, so that they stay in line
            columns.append([_num(here["rank_score"] if here else None) + count.ljust(room) for here, count in zip(mine, fewer)])
        lines += _table(header, [[m["model"]] + list(cells) for m, cells in zip(across, zip(*columns))])
        if any(cell.endswith(")") for column in columns for cell in column):
            lines.append("  A number in brackets after a figure: the sets at that width that scored the model, where it has no rank score in the others.")
        lines.append("  A standing against the other models at that width, not a measure of the pictures themselves.")

    # Where some set lacks a model, the figure over the shared sets differs from the mean over every set and stands beside it.
    like_for_like = headline["sets_common"] != headline["sets"]
    # In a run of one width the standings are over the same sets, and the mean over them is printed only for what it adds:
    # the figure over the shared sets, the figures of each judge alone, or means that are not the figures of the standings.
    if judges and (like_for_like or len(judges) > 1 or not means_repeat_standings(scores)):
        star = "*" if any(_starred(entry) for m in models for entry in [m] + m["by_judge"]) else ""
        lines += ["", f"All sizes together (the mean of a model's rank scores over every set that scored it; sets: how many, of {headline['sets']})"]
        each = judges if len(judges) > 1 else []  # the figures of one judge alone: with one judge they are the pooled ones
        rows = []
        for m in models:
            theirs = {j["judge"]: j for j in m["by_judge"]}
            row = [m["model"], _overall_text(m, star), _range_text(m["rank_score_interval"]), str(m["sets_scored"])]
            row += [_num(m["rank_score_common_sets"])] if like_for_like else []
            rows.append(row + [_overall_text(theirs.get(j), star) for j in each])
        lines += _table(["model", "all judges" if each else "mean", "95% interval", "sets"] + (["shared sets"] if like_for_like else []) + each, rows)
        if any(m["rank_score_interval"] for m in models) and "interval" not in said:  # a model with scores at two paintings, though at no one width
            lines.append(_interval_text(headline["paintings"], headline["draws"]))
        # A difference with no interval, in a run of two paintings or more: one of the two has scores at fewer than two of them.
        # (In a run of one painting no difference has an interval, and the line under the standings says so of all of them.)
        named = any(m["rank_score_interval"] for m in models) or headline["paintings"] >= 2
        lines += _open_pairs_text(headline["pairs"], {m["model"] for m in models if not m["rank_score_interval"]}, named, said, places=False)
        alone = []
        for pair in headline["pairs"]:
            for j in pair["by_judge"]:
                if j["judge"] in pair["reversed_by"]:
                    alone.append(f"    {j['judge']} alone has {pair['lower']} above {pair['higher']} ({_signed(j['difference'])})")
                elif pair["separated"] and j["separated"] is False:
                    alone.append(
                        f"    {pair['higher']} over {pair['lower']} is not separated on {j['judge']} alone "
                        f"({_signed(j['difference'])}, interval {_range_text(j['interval'], True)})"
                    )
        if alone:
            lines += ["  By one judge alone:"] + alone
        if star:
            lines.append(
                "  * no rank score in some of the sets: the figure rests on fewer sets than the list holds, and is not over the "
                "same sets as an unmarked figure. The report has the scores of each set."
            )
        if like_for_like and headline["sets_common"]:
            lines.append(
                f"  shared sets: the same mean over only the {headline['sets_common']} {'set' if headline['sets_common'] == 1 else 'sets'} in which every scored model has a rank score, "
                "where all models are compared on the same sets."
            )
        elif like_for_like:
            lines.append(
                "  shared sets: no set has a rank score for every scored model, so there is no figure that compares them all on "
                "the same sets."
            )

    partial: dict[str, list[str]] = defaultdict(list)
    for s in scores["combined"] + scores["sets"]:
        if s["partial_schedule"]:
            partial[f"{s['painting']}/{s['width']}"].append(s["judge"] or "all judges")
    if partial:
        lines += ["", f"Partial schedule (some pictures of different models were never compared): {len(partial)}"]
        lines += [f"  {where}: {', '.join(whose)}" for where, whose in partial.items()]
        lines.append(
            "  A rank score in these sets also depends on which opponents each picture met. Where a set was judged in pairs, go by "
            "its Bradley-Terry order (the order of its pictures in the report), which allows for how strong the opponents were."
        )

    lines += ["", "Completion and care (means; - is unknown)"]
    # With a plan, the planned grid stands beside what was run: a model with attempts never started is not read as a smaller run.
    plan = {m["model"]: m for m in planned["by_model"]} if planned and planned["attempts"] else None
    header = ["model"] + (["planned", "never started"] if plan is not None else []) + ["attempts", "complete", "incomplete", "refused", "error"]
    header += ["rejected first", "rejected", "nudges", "cut off", "calls", "seconds", "cost USD"]
    rows = []
    for c in scores["completion"]["by_model"]:
        mine = plan.get(c["model"]) if plan is not None else None  # a model on disk that the config does not plan has no figure
        row = [c["model"]] + ([str(mine["planned"]) if mine else "-", str(mine["not_started"]) if mine else "-"] if plan is not None else [])
        row += [str(c["attempts"]), percent_text(c["share_complete"]), str(c["incomplete"]), str(c["refused"]), str(c["error"])]
        row += [_num(c["mean_rows_rejected_first_submission"], 1), _num(c["mean_rows_rejected"], 1), _num(c["mean_nudges"], 1)]
        rows.append(row + [_num(c["mean_truncated_replies"], 1), _num(c["mean_model_calls"], 1), _num(c["mean_wall_seconds"], 0), _num(c["cost_usd"])])
    lines += _table(header, rows) if rows else ["  no attempts yet"]
    if rows and plan is not None:
        lines.append(
            "  planned: the attempts the config plans. never started: those of them with no record. attempts: those with a record. "
            "complete: the share of the attempts with a record that are complete."
        )
    if rows:
        lines.append("  nudges: reminders to finish that did not follow a cut-off reply. cut off: replies cut off by the output limit.")
    if total["refused_by_provider"] or total["refused_in_words"]:
        lines.append("  Of the refused: blocked by the provider (a platform block, not the model declining), and declined by the model in words")
        for c in scores["completion"]["by_model"]:
            if c["refused"]:
                unsaid = c["refused"] - c["refused_by_provider"] - c["refused_in_words"]
                lines.append(
                    f"    {c['model']}: {c['refused_by_provider']} blocked by the provider, {c['refused_in_words']} declined in words"
                    + (f", {unsaid} with no record of which" if unsaid else "")
                )

    if scores["peak_combined"]:
        lines += ["", "Peak size (the width judged best among a model's own complete sizes)"]
        for t in scores["peak_tally"]:
            best = [f"{o['width']} on {o['paintings']}" for o in sorted(t["outright"], key=lambda o: -o["paintings"]) if o["paintings"]]
            best += [f"{_widths_text(tie['widths'])} tied on {tie['paintings']}" for tie in t["tied"]]
            lacking = f"; in {t['larger_not_complete']} a larger width has no complete picture" if t["larger_not_complete"] else ""
            lines.append(f"  {t['model']}, {t['paintings']} paintings: {', '.join(best)}{lacking}")
        for p in scores["peak_combined"]:
            best = "tied between " + " and ".join(str(w) for w in p["best_widths"]) if p["tied"] else f"best at {p['best_width']}"
            lacking = f" ({_widths_text(p['larger_not_complete'])} not complete)" if p["larger_not_complete"] else ""
            by_width = ", ".join(f"{w['width']}: {_num(w['mean_rank'])}" for w in p["widths"])
            lines.append(f"  {p['painting']}  {p['model']}: {best}{lacking} (mean rank by width {by_width})")
        if any(p["larger_not_complete"] for p in scores["peak_combined"]):
            lines.append("  Where a larger width is not complete, the figure is the best of the sizes the model finished, not a peak.")

    repeating = [row for row in scores["pictures_by_width"] if row["under_half"]]
    if repeating:
        lines += ["", "Complete pictures with fewer than half their rows distinct (complete means every row was stored)"]
        lines += [f"  {row['model']} at {row['width']} wide: {row['under_half']} of {row['pictures']}" for row in repeating]
        lines.append("  For these the larger sizes partly measure how much of the grid was filled, not only what the model recalls.")

    if scores["judge_vs_person"]:
        lines += ["", "Judge versus person"]
        for row in scores["judge_vs_person"]:
            c = row["combined"]
            line = f"  {row['painting']}/{row['width']}  {row['judge']} v {row['person']}: tau-b {_num(c['tau_b'])} on the judge's combined order"
            if row["repeats"]:
                line += f", {row['repeats_within_one_adjacent_swap']} of {row['repeats_compared']} repeats within one swap of neighbours"
            else:
                line += f", swapped: {_pairs_text(c['swapped_pairs'])}"
            lines.append(line)
            for r in row["repeats"]:
                lines.append(f"      repeat {r['repeat']}: tau {_num(r['tau'])}, swapped: {_pairs_text(r['swapped_pairs'])}")

    if scores["judges"]:
        lines += ["", "Judges"]
        for j in scores["judges"]:
            line = f"  {j['judge']}: {j['calls']} calls in use"
            if j["contestants"]:
                line += f"; itself a contestant ({j['model']}, entered as {', '.join(j['contestants'])})"
            if j["calls_reporting_api_retries"]:
                line += f"; {j['calls_with_api_retries']} of {j['calls_reporting_api_retries']} calls needed API retries ({j['api_retries']} retries)"
            if j["calls_with_order"]:
                line += (
                    f"; ranked the picture shown first as best in {j['first_shown_best']} of {j['calls_with_order']} calls "
                    f"({j['first_shown_expected']:.1f} expected by chance)"
                )
            lines.append(line)
        lines.append("  Labels and the order shown are drawn at random for every call. A call that failed and was made again leaves no trace.")
        if not counts["human_rankings"]:
            waiting = counts["human_rankings_pending"]
            lines.append("  No person has ranked a sheet" + (f" ({waiting} made and waiting)" if waiting else "") + ": the judges are unchecked.")
        agreement = scores["judge_agreement"]
        if agreement["between_orders"]["sets"] or agreement["within"]:
            between, single = agreement["between_orders"], agreement["between_calls"]
            lines.append("  Agreement (Kendall's tau):")
            if between["sets"]:
                lines.append(
                    f"    between the judges' averaged orders: {_num(between['tau_b'])} over {between['sets']} sets (lowest {_num(between['lowest'])})"
                )
            if single["pairs"]:
                lines.append(f"    between single calls of different judges: {_num(single['tau'])} over {single['pairs']} pairs of calls")
            for row in agreement["within"]:
                lines.append(f"    between {row['judge']}'s own repeats: {_num(row['tau'])} over {row['pairs']} pairs of calls")

    measured = [row for row in scores["own_model"] if row["difference"] is not None]
    if measured:
        lines += ["", "Own model (a contestant's rank score from the judge of its own model id, minus the same from the other judges)"]
        for row in measured:
            lines.append(
                f"  {row['judge']} on {row['contestant']}: {_signed(row['difference'], 3)} ({_num(row['own_score'])} v {_num(row['others_score'])} "
                f"over {row['sets']} sets; interval {_range_text(row['interval'], True, 3)})"
            )
        lines.append("  A difference does not say which judge is off.")

    if scores["judge_vs_judge"]:
        lines += ["", "Judge versus judge"]
        for row in scores["judge_vs_judge"]:
            lines.append(
                f"  {row['painting']}/{row['width']}  {row['judge_a']} v {row['judge_b']}: tau-b {_num(row['tau_b'])}, "
                f"swapped: {_pairs_text(row['swapped_pairs'])}, tied pairs: {len(row['tied_pairs'])}"
            )

    favour = [row for row in scores["own_family"] if row["difference"] is not None]
    # With only judges of one family there is no baseline: say so, where a bare gap would read as "no favour found".
    no_baseline = [row for row in scores["own_family"] if row["same_family_judges"] and len(row["same_family_judges"]) == len(scores["own_family"]) - 1]
    if favour or no_baseline:
        lines += ["", "Own family (the judge's mean rank score for contestants of its own family, minus that of the judges of other families)"]
        for row in favour:
            lines.append(
                f"  {row['judge']} ({row['family']}): {row['difference']:+.2f} "
                f"({_num(row['own_mean'])} v {_num(row['others_mean'])} over {row['scores_compared']} scores)"
            )
        for row in no_baseline:
            lines.append(f"  {row['judge']} ({row['family']}): not measured, every other judge ({', '.join(row['same_family_judges'])}) is of its own family")

    if scores["not_complete"]:
        lines += ["", f"Not complete: {len(scores['not_complete'])}"]
        lines += [f"  {a['attempt']}: {_ending_text(a)}" for a in scores["not_complete"]]

    if planned and planned["not_started"]:
        lines += ["", f"Never started (planned by the config, no record in the run, no reason recorded): {len(planned['not_started'])}"]
        lines += [f"  {a['attempt']}" for a in planned["not_started"]]

    if scores["retried"]:
        lines += ["", f"Run again (an earlier run of the attempt was replaced): {len(scores['retried'])}"]
        for a in scores["retried"]:
            lines.append(f"  {a['attempt']}: now {a['state']} ({a['stop']}); before: {', then '.join(_ending_text(e) for e in a['earlier'])}")

    waiting = [g for g in scores["grids"] if g["not_judged"]]
    if waiting:
        again = " Running the judge stage again will rank them." if any(len(g["complete"]) > 1 for g in waiting) else ""
        lines += ["", f"Not judged yet (complete attempts that no judgement in use ranks): {sum(len(g['not_judged']) for g in waiting)}.{again}"]
        for g in waiting:
            alone = "" if len(g["complete"]) > 1 else " (alone at this size: a ranking needs two complete attempts)"
            lines.append(f"  {g['painting']}/{g['width']}: {_names_text(g['not_judged'])}{alone}")

    failed = [f for f in scores["failed_judgements"] if f["status"] not in ("stale", "superseded")]
    if failed:
        lines += ["", f"Failed judgements (not scored): {len(failed)}"]
        lines += [f"  {f['path']}" for f in failed]
    reasons = (
        ("picture", "A picture they ranked was replaced or is no longer complete."),
        ("reference", "The painting's reference image has been replaced since they were made."),
    )
    stale = [f for f in scores["failed_judgements"] if f["status"] == "stale"]
    for reason, what in reasons:
        these = [f for f in stale if f["stale_reason"] == reason]
        if these:
            lines += ["", f"Stale judgements (not scored): {len(these)}. {what}"] + _by_directory(these)
            lines.append("  Running the judge stage again remakes the calls it still plans.")
    superseded = [f for f in scores["failed_judgements"] if f["status"] == "superseded"]
    if superseded:
        lines += ["", f"Superseded judgements (not scored): {len(superseded)}. Their set is now judged the other way (whole, or in pairs)."]
        lines += _by_directory(superseded)
        lines.append("  The judge stage no longer makes these calls, so it does not replace them. The files are left on disk.")
    if scores["reference_unchecked"]:
        lines += ["", f"Reference not checked: {', '.join(scores['reference_unchecked'])}. The painting's metadata or reference image was not found,"]
        lines.append("  so its judgements could not be held against the image they were made with. Name the paintings directory in the config,")
        lines.append("  or fetch a missing image with: python scripts/fetch_originals.py <painting>")
    return "\n".join(lines) + "\n"

# Writing Style Summary for Density4Drug

This guide summarizes the shared writing patterns in the four reference papers:
[TriProRep](https://arxiv.org/abs/2605.22133),
[Riemannian MeanFlow](https://arxiv.org/abs/2602.07744),
[BioEmu-CV](https://arxiv.org/abs/2507.07390), and
[INDIBATOR](https://arxiv.org/abs/2602.01815).
It is adapted to the abstract, introduction, and related work of Density4Drug.

## Core principle

State the paragraph's main point in its first sentence. Use the remaining
sentences to explain, support, qualify, or connect that point. A reader who sees
only the first sentence of each paragraph should still recover the paper's
argument.

## Paragraph structure

Aim for about five sentences per paragraph:

1. State the main claim.
2. Give the most important context or evidence.
3. Add a mechanism, example, or comparison.
4. Explain the consequence, limitation, or unresolved issue.
5. Bridge naturally to the next paragraph.

The final sentence should advance the argument. It should not merely repeat the
first sentence.

## Sentence style

- Prefer roughly 12--20 words per sentence; treat 25 words as a soft ceiling.
- Express one main idea per sentence.
- Put the grammatical subject and main verb early.
- Prefer direct verbs: “retains,” “tests,” “improves,” and “reveals.”
- Use a second sentence when a contrast, list, or qualification becomes dense.
- Keep technical terms when they add precision; do not replace them with longer
  synonyms merely to sound formal.
- Use explicit nouns when pronouns such as “this” or “it” have unclear referents.
- Qualify empirical claims precisely with terms such as “point estimate,”
  “mean,” “subset,” or “predicted.”

## Abstract pattern

The abstract should form one compact argument:

1. Establish the broader setting.
2. Identify what existing representations omit.
3. Explain why that missing information may matter.
4. State the research question.
5. Introduce the method and pre-training data.
6. Name the downstream evaluations.
7. Report the strongest quantitative results.
8. State an important limitation or trade-off.
9. End with the narrowest conclusion supported by the evidence.

The abstract is an exception to the five-sentence paragraph preference. It can
contain more sentences because each sentence should perform one step in the
argument.

## Introduction pattern

Use a six-paragraph progression:

1. **Setting:** reusable representations matter for protein machine learning.
2. **Limitation:** fitted coordinates omit part of the crystallographic evidence.
3. **Prior evidence and gap:** density helps individual tasks, but transferable
   density representations remain less studied.
4. **Method:** introduce Density4Drug, its inputs, data, and objective.
5. **Evaluation:** explain the predictive and generative transfer settings.
6. **Findings:** give the main gains and acknowledge the pose-quality trade-off.

The contribution list should add scannability, not introduce new claims. Keep
its items parallel: introduce the representation, evaluate affinity transfer,
and evaluate generative transfer.

## Related-work pattern

Give each topic one focused paragraph. Start by defining why the topic matters,
then summarize representative approaches by conceptual group rather than paper
by paper. End by stating the specific difference addressed by Density4Drug.

A useful five-sentence template is:

1. Define the topic or its relevance.
2. Describe the first major approach family.
3. Describe the second family or recent development.
4. Identify the limitation relevant to this paper.
5. State how the present work differs.

Keep the comparison narrow. Avoid generic claims that Density4Drug is simply
“better” or “more comprehensive.” Name the concrete distinction, such as joint
coordinate--density pre-training, frozen transfer, or leakage-controlled
evaluation.

## Transitions

Transitions should express a logical relation rather than announce a section.
Useful patterns include:

- **Consequence:** “This makes the retained input information increasingly important.”
- **Narrowing:** “This evidence may be especially relevant near a ligand.”
- **Gap:** “These studies establish utility, but not transfer across tasks.”
- **Response:** “We address this gap with a coordinate-and-density encoder.”
- **Qualification:** “At the same time, pose checks reveal a remaining gap.”

Avoid empty transitions such as “Moreover,” “Furthermore,” or “It is worth
noting that” unless the logical relationship is otherwise clear.

## Final editing checklist

- Does the first sentence state the paragraph's main point?
- Does every later sentence support that point or lead to the next paragraph?
- Can any sentence be split without losing its logical connection?
- Does each sentence add information not already stated nearby?
- Are method and result claims supported by the current experiments?
- Are point estimates distinguished from statistically conclusive gains?
- Are limitations reported where they affect the interpretation?
- Are terminology, capitalization, and dataset names consistent throughout?

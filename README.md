# glue-benchmark
take on how close we can get to a flexible glue-benchmark where it matters

# Why
Apart from measuring the progress of research in NLP and NLP transfer learning, the Glue collection offers a good and varied set of low level NLP capabilities which can be used in a variety of higher level solutions. For instance, in large text & news corpora discerning entailment is key to reducing the volume of inputs as well as identifying truly new information

| Index | Description | Inputs | Target | Metric | SOTA | Best here |
|--|--|--|--|--|
| CoLA | Linguistic acceptability | Sent1 & 2 | Binary | Accuracy | 75% | 76% |
| SST-2 | Sentiment analysis | Sent1 | Regression | Correlation | 97% | 91% |
| MRPC | Sentence equivalence | Sent1, Sent2 | Binary | Accuracy | | |
| STS-B | Meaning similarity | Sent1, Sent2 | Regression | Correlation | | |
| QQP | Quora Question Pairs, Question equivalence (binary) | Sent1, Sent2 | Binary | Accuracy | | |
| MNLI-m | Matched-Textual entailment (meaning of one fragment is contained in another). Targets: entailment (repeat meaning), contradiction (opposite meaning), or neutral (not relevant) | Sent1, Sent2 | entailment, contradiction, or neutral | Accuracy, F1 | | |
| MNLI-mm | Same as above, Mismatched- refers to mismatch between trained domains and test domains | Sent1, Sent2 | entailment, contradiction, or neutral | Accuracy, F1 | | |
| QNLI | Stanford Question Answering Dataset (SQuAD), determine what is the answer and if the answer is available in the paragraph reference | Question, Paragraph | Binary & Sequence with the answer | Accuracy | | |
| RTE | Recognising Textual Entailment | Sent1, Sent2 | entailment, contradiction, or neutral | Accuracy, F1 | | |
| WNLI | Winograd Schema, Pronoun ambiguity where the answer requires world knowledge and not only grammatical context | Sent1, Sent2 | entailment, contradiction, or neutral | Accuracy, F1 | | |
| AX | DiagnosticMain, Different entailment relationships of arbitrary size predominantly for diagnostic purposes | Sent1, Sent2 | entailment, contradiction, or neutral | Accuracy, F1 |  |  . |
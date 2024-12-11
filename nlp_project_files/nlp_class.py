"""
filename: nlp_class.py

includes an NLP class that processes and cleans text files into a meta dictionary of information about the text and uses
the stored data to create visualizations
"""

import string
from collections import Counter
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict


class NLP:
    def __init__(self):
        self.texts = {}
        self.stop_words = set()
        self.data = {
            "word_count": {},
            "sentiment": {},
            "clean_text": {},
            "vocabulary_richness": {}
        }

    def load_text(self, filename, parser=None, label=None):
        """
        processes and cleans a text file.
        """
        processed_lines = []

        with open(filename, 'r') as file:
            for line in file:
                # remove whitespace and lowercase words
                line = line.strip().lower()
                # remove punctuation
                line = line.translate(str.maketrans('', '', string.punctuation))
                words = line.split()

                # remove stopwords
                filtered_words = [word for word in words if word not in self.stop_words]

                # apply custom parser if provided
                if parser:
                    filtered_words = parser(filtered_words)

                processed_lines.append(" ".join(filtered_words))

        # store clean text in data dictionary
        self.texts[filename] = processed_lines
        self.data["clean_text"][label] = processed_lines

    def load_stop_words(self, stopfile):
        """
        loads a list of stop words from stopwords file
        """
        with open(stopfile, 'r') as file:
            self.stop_words = set(file.read().splitlines())

    def compute_word_counts(self):
        """
        computes word counts and stores them in self.data
        """
        for label, processed_lines in self.data["clean_text"].items():
            all_words = " ".join(processed_lines).split()
            word_counts = Counter(all_words)
            self.data["word_count"][label] = word_counts

    def compute_sentiments(self):
        """
        computes sentiment scores for each unique word and stores them in self.data
        """
        word_sentiments = {}

        for label, processed_lines in self.data["clean_text"].items():
            all_words = " ".join(processed_lines).split()
            for word in all_words:
                if word not in word_sentiments:
                    # compute sentiment only for unique words
                    word_sentiments[word] = TextBlob(word).sentiment.polarity

            # store the sentiment score of each word in the file
            self.data["sentiment"][label] = {word: word_sentiments[word] for word in all_words}

    def compute_vocabulary_richness(self):
        """ Computes the vocabulary richness (TTR) for each text and stores it in self.data. """
        vocabulary_richness = {}

        for label, processed_lines in self.data["clean_text"].items():
            all_words = " ".join(processed_lines).split()
            unique_words = set(all_words)
            ttr = len(unique_words) / len(all_words) if all_words else 0
            vocabulary_richness[label] = {
                "TTR": ttr,
                "Unique Words": len(unique_words),
                "Total Words": len(all_words)
            }

        self.data["vocabulary_richness"] = vocabulary_richness

    def wordcount_sankey(self, k=5, category_mapping=None):
        """
        Generating a sankey that displays political standing (left, right, center) pointing towards
        the top k most frequent words in each political category.
        If words overlap, they merge to one word node
        """
        if not self.data["word_count"]:
            raise ValueError("Word counts not computed. Please run compute_word_counts().")

        if category_mapping is None:
            raise ValueError("Category mapping is required to group files.")

        # aggregating word ct by category
        category_word_counts = defaultdict(Counter)
        for file_label, counts in self.data["word_count"].items():
            category = category_mapping.get(file_label, "Unknown")
            category_word_counts[category].update(counts)

        # retrieving top k words per category
        category_top_words = {
            category: [word for word, _ in counts.most_common(k)]
            for category, counts in category_word_counts.items()
        }

        # creating lists for flows, sources, and targets
        flows, sources, targets = [], [], []

        # adding connection from each category to respective top words
        for category, top_words in category_top_words.items():
            for word in top_words:
                flow = category_word_counts[category][word]
                sources.append(category)
                targets.append(word)
                flows.append(flow)

        # dataframe of src, targ, val
        df = pd.DataFrame({"source": sources, "target": targets, "value": flows})

        # generating diagram
        self.make_sankey(df, 'source', 'target', 'value')

    def make_sankey(self, df, src, targ, vals, **kwargs):
        """
        creating a sankey using plotly.
        df - DataFrame
        src - Source node column
        targ - Target node column
        vals - Link values (thickness)
        """
        # map labels to ints
        df, labels = self._code_mapping(df, src, targ)

        # create figure
        link = {'source': df[src], 'target': df[targ], 'value': df[vals]}
        node = {'label': labels, 'thickness': kwargs.get('thickness', 50), 'pad': kwargs.get('pad', 50)}

        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        fig.show()

    def _code_mapping(self, df, src, targ):
        """
        maps labels in src and targ columns to integers
        """
        # get labels
        labels = sorted(list(set(df[src]) | set(df[targ])))
        # crate label code mapping
        mapping = {label: i for i, label in enumerate(labels)}
        # sub codes for labels in df
        df = df.replace({src: mapping, targ: mapping})
        return df, labels

    def bubble_chart(self, file_article_mapping):
        """
        Creates a bubble chart showing frequency and sentiment for all articles, now reflecting individual word sentiments.
        """

        data = []
        # using meta dictionary information
        for label, article_type in file_article_mapping.items():
            word_counts = self.data["word_count"].get(label, {})
            word_sentiments = self.data["sentiment"].get(label, {})

            for word, count in word_counts.items():
                sentiment = word_sentiments.get(word, 0)
                data.append({
                    "Word": word,
                    "Frequency": count,
                    "Sentiment": sentiment,
                    "Article Type": article_type
                })

        # convert to DataFrame
        df = pd.DataFrame(data)

        # generate bubble chart
        fig = px.scatter(
            df,
            x="Frequency",
            y="Sentiment",
            size="Frequency",
            color="Article Type",
            hover_name="Word",
            title="Word Analysis for Articles",
            labels={"Frequency": "Word Frequency", "Sentiment": "Sentiment Polarity"}
        )

        fig.show()

    def sunburst_vocabulary_richness(self, file_article_mapping):
        """ Creates a sunburst plot to analyze vocabulary richness by article type. """
        data = []

        for label, article_type in file_article_mapping.items():
            richness = self.data["vocabulary_richness"].get(label, {})
            data.append({
                "Article Type": article_type,
                "Label": label,
                "TTR": richness.get("TTR", 0),
                "Unique Words": richness.get("Unique Words", 0),
                "Total Words": richness.get("Total Words", 0)
            })

        # Convert to df
        df = pd.DataFrame(data)

        # Generate sunburst Plot
        fig = px.sunburst(
            df,
            path=["Article Type", "Label"],
            values="Total Words",
            color="TTR",
            hover_data={"Unique Words": True, "TTR": True},
            title="Vocabulary Richness Analysis (TTR)",
            labels={"Total Words": "Total Word Count"}
        )

        fig.show()

    def show_data(self):
        """
        prints the current state of the data dictionary in a readable format
        """
        print(self.data)
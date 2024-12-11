"""
filename: nlp.main.py

uses NLP class to analyze articles with different political leanings
"""

from nlp_class import NLP

if __name__ == "__main__":
    nlp = NLP()

    # load stop words
    nlp.load_stop_words("stopwords.txt")

    # load and process the text files
    nlp.load_text("left_newyorktimes.txt", label="left_1")
    nlp.load_text("left_abc.txt", label="left_2")
    nlp.load_text("left_cnn.txt", label="left_3")
    nlp.load_text("right_fox.txt", label="right_1")
    nlp.load_text("right_cbn.txt", label="right_2")
    nlp.load_text("right_dispatch.txt", label="right_3")
    nlp.load_text("center_bbc.txt", label="center_1")
    nlp.load_text("center_cnbc.txt", label="center_2")
    nlp.load_text("center_forbes.txt", label="center_3")

    # compute word counts and sentiments
    nlp.compute_word_counts()
    nlp.compute_sentiments()

    # Mapping labels to categories
    category_mapping = {
        "left_1": "Left", "left_2": "Left", "left_3": "Left",
        "right_1": "Right", "right_2": "Right", "right_3": "Right",
        "center_1": "Center", "center_2": "Center", "center_3": "Center"
    }

    # Generating sankey diagram, maximum word nodes are k * 3 (top k word frequencies * 3 categories)
    # If categories share top words, then they will share the same word node
    nlp.wordcount_sankey(k=10, category_mapping=category_mapping)

    # generate bubble chart
    file_article_mapping = {
        "left_1": "Left",
        "left_2": "Left",
        "left_3": "Left",
        "right_1": "Right",
        "right_2": "Right",
        "right_3": "Right",
        "center_1": "Center",
        "center_2": "Center",
        "center_3": "Center"
    }
    nlp.bubble_chart(file_article_mapping)


    nlp.compute_vocabulary_richness()

    # Generate sunburst plot
    nlp.sunburst_vocabulary_richness(file_article_mapping)

    # displays dictionary data
    nlp.show_data()

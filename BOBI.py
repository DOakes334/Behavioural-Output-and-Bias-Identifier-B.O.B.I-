
import spacy
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    console.print(
        "[bold red]Error: spaCy model 'en_core_web_md' not found.[/bold red]"
    )
    console.print("Please run: python -m spacy download en_core_web_md")
    exit()

try:
    nltk.data.find("corpora/wordnet.zip")
except nltk.downloader.DownloadError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4.zip")
except nltk.downloader.DownloadError:
    nltk.download("omw-1.4")

def generate_subtopics(topic: str) -> list[str]:
    """
    Generates 3 distinct subfields/related concepts for a given topic using NLTK WordNet.
    Falls back to generic sub-aspects if no hyponyms are found.
    """
    subtopics = set()
    # Replace spaces with underscores for WordNet compatibility
    formatted_topic = topic.replace(" ", "_")
    
    synsets = wordnet.synsets(formatted_topic)
    
    if synsets:
        # Prioritize hyponyms (more specific terms)
        for syn in synsets:
            for hyponym in syn.hyponyms():
                for lemma in hyponym.lemmas():
                    subtopics.add(lemma.name().replace("_", " ").title())
                    if len(subtopics) >= 10:  # Collect a good number to choose from
                        break
                if len(subtopics) >= 10:
                    break
            if len(subtopics) >= 10:
                break

    # If not enough subtopics found, add hypernyms (broader terms)
    if len(subtopics) < 3 and synsets:
        for syn in synsets:
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    subtopics.add(lemma.name().replace("_", " ").title())

    # Fallback to generic sub-aspects if still not enough subtopics
    if len(subtopics) < 3:
        generic_subtopics = [
            f"Benefits of {topic.title()}",
            f"Challenges in {topic.title()}",
            f"The Future of {topic.title()}",
            f"Applications of {topic.title()}",
            f"Ethical concerns of {topic.title()}",
        ]
        subtopics.update(generic_subtopics)

    return list(subtopics)[:3]

def analyze_text(text: str, topic: str) -> dict:
    """
    Analyzes a given text for multiple psycholinguistic metrics.
    """
    # Create spaCy and TextBlob objects
    doc = nlp(text)
    blob = TextBlob(text)

    # METRIC 1: Sentiment & Subjectivity
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    # METRIC 2: Cognitive Load
    sentences = list(doc.sents)
    num_sentences = len(sentences) if sentences else 1
    num_words = len([token for token in doc if not token.is_punct])
    avg_sentence_length = num_words / num_sentences

    # Lexical diversity (Type-Token Ratio)
    unique_words = len(set(token.text.lower() for token in doc if not token.is_punct))
    lexical_diversity = unique_words / num_words if num_words > 0 else 0

    # METRIC 3: "Ego Score"
    first_person_pronouns = len([token for token in doc if token.lemma_.lower() in ["i", "me", "my"]])
    ego_score = first_person_pronouns / num_words if num_words > 0 else 0

    # METRIC 4: Implicit Bias (Vector Cosine Similarity)
    topic_token = nlp(topic)[0]
    positive_words = [nlp("good")[0], nlp("logic")[0], nlp("safe")[0]]
    negative_words = [nlp("bad")[0], nlp("chaos")[0], nlp("danger")[0]]

    adjectives = [token for token in doc if token.pos_ == "ADJ"]
    bias_score = 0
    if adjectives and topic_token.has_vector:
        pos_sims = [adj.similarity(pos_word) for adj in adjectives for pos_word in positive_words if adj.has_vector]
        neg_sims = [adj.similarity(neg_word) for adj in adjectives for neg_word in negative_words if adj.has_vector]
        
        avg_pos_sim = sum(pos_sims) / len(pos_sims) if pos_sims else 0
        avg_neg_sim = sum(neg_sims) / len(neg_sims) if neg_sims else 0
        bias_score = avg_pos_sim - avg_neg_sim # Positive score -> positive bias, Negative score -> negative bias

    return {
        "Polarity": polarity,
        "Subjectivity": subjectivity,
        "Avg Sentence Length": avg_sentence_length,
        "Lexical Diversity": lexical_diversity,
        "Ego Score": ego_score,
        "Implicit Bias": bias_score,
    }

def main():
    """
    Main function to run the BOBI application.
    """
    console.print(Panel("[bold magenta]Welcome to B.O.B.I.[/bold magenta]\n[cyan]Behavioral Output & Bias Identifier[/cyan]", title="B.O.B.I.", border_style="green"))
    
    main_topic = Prompt.ask("[bold]Please enter the main topic you want to discuss[/bold]")
    
    console.print(f"\n[bold]Generating subtopics for '{main_topic}'...[/bold]")
    subtopics = generate_subtopics(main_topic)
    
    if not subtopics:
        console.print("[bold red]Could not generate subtopics. Exiting.[/bold red]")
        return

    console.print(f"Found subtopics: [yellow]{', '.join(subtopics)}[/yellow]")
    
    analyses = {}
    for subtopic in subtopics:
        console.print(Panel(f"Please write a short paragraph about: [bold cyan]{subtopic}[/bold cyan]", border_style="blue"))
        user_text = Prompt.ask(f"Your thoughts on [cyan]{subtopic}[/cyan]")
        analyses[subtopic] = analyze_text(user_text, subtopic)

    # Display results
    console.print("\n\n[bold underline green]Psycholinguistic Profile[/bold underline green]")
    for subtopic, results in analyses.items():
        report = (
            f"Polarity: {results['Polarity']:.2f} "
            f"| Subjectivity: {results['Subjectivity']:.2f}\n"
            f"Cognitive Load (Avg Sentence Length): {results['Avg Sentence Length']:.2f} words\n"
            f"Lexical Diversity (Unique/Total Words): {results['Lexical Diversity']:.2f}\n"
            f"Ego Score (Self-Focus): {results['Ego Score']:.3f}\n"
            f"Implicit Bias Score: {results['Implicit Bias']:.3f}"
        )
        console.print(Panel(report, title=f"Analysis for '{subtopic}'", border_style="yellow"))

if __name__ == "__main__":
    main()
# Assignment 0: Getting Started with the pre-requisites 

Hello, mentee! This assignment will guide you through all the necessary pre-requisites for your winter project. You are however, advised to explore more aspects apart from whatever covered in this assignment. 

## Some ID's that you need 

- Github ID: [Follow this hyperlink in case you need guidance to set up your Github account](https://medium.com/@e_idiaghe/how-to-setup-your-github-student-pack-b3b00562aa5b) 
(Note: It is recommended to use your Github ID for Authorization to websites wherever possible.)
- Discord ID: We shall shift our mode of communication to Discord in future. 
- And ofcourse, a Reddit Account. You are advised to go through some good subreddits related to your technical interests and recommend some in your submission file! 


## Pen and paper assignment 

Watching youtube videos, reading articles or reading code documentation straightaway would be a boring approach. Instead, we would like to motivate the concepts of Natural Language Processing through a simple written assignment to understand a simpler version of the project. 

Below are a set of questions for you to attempt, you are required to submit a pdf of your solutions through github. More details will be outlined in the "Submission Guidelines" section of the assignment. 

#### Consider these three posts for the entirety of the pen and paper assignment 
#### Post A: "Cats are cute and funny" 
#### Post B: "Dogs are funny animals" 
#### Post C: "Cats and dogs rarely get along" 

### 1. Implementing Bag-of-words 
- The above three posts are the entire body of text that you have for analysis. This whole body is called __corpus__. This corpus is divided (by the convenience of the structure) into three parts. Construct the __vocabulary__ of the corpus, i.e. a list of all _unique_ words in the corpus. (take all words to be lowercase)

- In some more technical sense this list or "bag" of the unique words constructed is called a **word vector**, keep in mind the order of the words in your vector (ordering is an important property of a vector). The number of entities (in this case words) is the dimension of the vector. What is the dimension of the bag-of-words for this corpus? 

- For each post, construct a **count vector** (also called Bag-of-Words vector) where each entry is the number of times the corresponding vocabulary word appears in that post. 

- Stack the three vectors into a matrix. What is the shape of this matrix? Can you highlight some aspects of the matrix (rows and columns) with relation to the word vector of the corpus? (Hint: You may want to use the fact that the corpus is divided into three posts) 

### 2. Designing a simple model for classification 

In this segment, we will design a simple probability-based model for classifying a post into either a cat-type or a dog-type post. We will make use of the Bag-of-Words vector above to accomplish this task. 

- Suppose we scan a post word-by-word. Each word has $p$ probability of being the word "cat", i.e. $$P(cat) = p$$ independent of others. If a post has 10 words, what is the probability that it contains no occurrence of “cat”? What is the probability it contains **atleast one** occurence of the word "cat"? If this is so then the post is classified as "cat-type" post. Obtain the expression for probability of a post of length $L$ being a "cat-type" post.

- From the count vector constructed, now make a **Probability Vector** where each entry of the vector is the statistical probability of that word occuring in the whole corpus. For example, if your count vector is $[1,2,3,4]$ then its probability vector would be $[0.1, 0.2, 0.3, 0.4]$, for the obtained probability vector, substitute $p$ of the previous question and calculate the probability of post A being a cat-type post following the instructions outlined in the previous part (note that post A contains 5 words, also note that you have to report the **theoretical** probability of post A being a cat-type post, and is independent of the fact whether post A already contains the word "cat" or not)
-  Using the fact that Post A and Post C are cat-type posts and using conditional probability P(A given B): 
$$ P(A|B) = \frac{P(A \cap B )}{P(B)}$$ 
Calculate the probability that a post contains the word "cute" given it is a cat-type post. What does this tell you about cat-type posts? You may want to use the Bayes Theorem: 
$$ P(B|A) = \frac{P(A|B)*P(B)}{P(A)}$$ 
to calculate the probability of the post being a "cat-type" given it contains the word "cute". 

### 3. Designing a simple model for Optimising Upvotes 

In this segment, we will design a simple Upvote Optimisation Algorithm and try to find the constraints for getting a post that can maximise the upvote count. 

- Suppose upvotes can be approximated (hypothetically) as a quadratic function of post length 𝐿 $$U(L) = -\frac{1}{20}L^2 + 3L$$ 
Solve $U'(L) = 0$ to find the optimum length of posts for getting maximum upvotes. Using the second derivative test show that it is indeed the maximum. 

- In the previous question you obtained the expression for the probability of a post of length $L$ being a cat-type post. There is actually a relation between cat-type post and the number of upvotes they get. Let us define the probability function as $P(L,p)$ where notations are as outlined previously. Let the new function be: $$G(L,p) = P(L,p).U(L)$$ 
Finding the optimum $L$ and $p$ would be a cumbersome task, so instead of the usual derivative-based algorithm, can you suggest some other ways to find the same? (Hint: Consider $L$ tending to infinity)


## Resources for tools and Python libraries

### Week 0 Resources

- Google Colab:-
https://www.youtube.com/watch?v=inN8seMm7UI&feature=youtu.be
Get started with Google Colaboratory (Coding TensorFlow) (A short video to introduce you to the place where you’ll code your project)


- GitHub-
https://youtu.be/r8jQ9hVA2qs?si=n2FwuqpAz4Dg5T3a
 A brief introduction to Git for beginners, focusing on GitHub (watch until “Basic Git concepts”). 

- https://www.youtube.com/watch?v=tlu5e0TxSzo	 
How to Upload Files and Folders to GitHub: GitHub for Beginners (watch until “Using GitHub UI to upload files”)


- Basic Python:-https://www.w3schools.com/python/  ( A good resource to learn with examples, the best way though, no need to mug up all the syntax, just have a basic idea of where it can be used so that you can go back and use it.)

- Numpy-  https://www.w3schools.com/python/numpy/default.asp

- Pandas-https://www.w3schools.com/python/pandas/default.asp

- Matplotlib: https://www.w3schools.com/python/matplotlib_intro.asp


## Coding Assignment 

After going through all the resources, attempt this assignment: 
- [Google Colab Assignment](https://colab.research.google.com/drive/1rLL6HW72S27K2Q_xcXgt3yaEj1LFlLwC?usp=sharing) 

### Basics of Machine Learning 

- Watch videos #3 to #14 from the playlist below (:) dw these are short videos)
https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=G_tJsm-zS6NJK0mq	


- Now, moving to the core of our work, the key resources we’ll be using for our project
Basic NLP (Text Preprocessing):- 
https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e

#### Diving Deep into Text Representation in NLP

Imagine distilling a sentence into a mere collection of words, disregarding grammar and word order. This is the essence of the Bag-of-Words (BoW) model, which emphasises the mere occurrence of words. It’s a straightforward way to tokenise text, but it doesn’t capture context.
You can find out more here: [Bag of Words](https://ayselaydin.medium.com/4-bag-of-words-model-in-nlp-434cb38cdd1b)

Imagine trying to categorize a group of objects where each object belongs to a unique category. One-hot encoding is akin to assigning a unique ID to each word in your text, where each ID contains just one slot marked “1” and all other slots are marked “0.”
You can find out more here: [One Hot Encoding](https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148)

TF-IDF enhances BoW by assigning weights to terms according to their relevance. It emphasises important words while minimising the influence of frequent, less significant ones. TF (Term Frequency) measures how often a word appears in a document, while IDF (Inverse Document Frequency) gauges the word’s rarity across documents.
You can find out more here: [TF-IDF Vectorizer](https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%20%2D%20Inverse%20Document%20Frequency%20(TF%2DIDF)%20is,%2C%20relative%20to%20a%20corpus)

CBOW, a member of the Word2Vec family, predicts a target word by leveraging its surrounding context. It's akin to inferring a missing word in a sentence by analysing the neighbouring words, thereby effectively capturing the underlying semantic relationships.
You can find out more here: [CBOW](https://ayselaydin.medium.com/11-word2vec-approaches-word-embedding-in-nlp-538478c14b37)

**Sentiment Analysis**
Sentiment classification models analyze text to predict its underlying sentiment based on the words and phrases employed. These sentiments are typically categorised into three distinct types: Positive Sentiment, which denotes a favourable opinion or contentment; Negative Sentiment, signalling dissatisfaction, critique, or adverse views; and Neutral Sentiment, where the text conveys no discernible sentiment or remains ambiguous.

Explore more here:
[Sentiment Analysis](https://medium.com/analytics-vidhya/nlp-getting-started-with-sentiment-analysis-126fcd61cc4a)

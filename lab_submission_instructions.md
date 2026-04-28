# Lab Submission Instructions

---

## Student Details

**Name of the team on GitHub Classroom:** 202509-j

**Team Member Contributions:**

**Member 1**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 147473 |
| **Name:**                                                                                          | Abdullahi Amin Hassan |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Completed both notebooks individually, topic modelling using LDA and sentiment analysis using a supervised classifier. Built and deployed a Streamlit interface for real-time topic and sentiment prediction. Learned how LDA discovers latent topics in large text corpora and how TF-IDF combined with a classifier can effectively predict sentiment from hotel reviews. |

## Scenario

Your client, a university, is seeking to enhance their qualitative analysis of
student course evaluations collected from students. They have provided you
with a dataset containing student course evaluation for two courses in the
Business Intelligence Option. The two courses are:

- BBT 4106: Business Intelligence I
- BBT 4206: Business Intelligence II

The client wants you to use Natural Language Processing (NLP) techniques to identify
the key topics (themes) discussed in the course evaluations. They would also like to
get the sentiments (positive, negative, neutral) of each theme in the course evaluation.

Lastly, the client would like an interface through which they can provide input in the
form of new textual data (one student's textual evaluation at a time) and the output
expected is:

1. The topic (theme) that the new textual data is talking about.
2. The sentiment (positive, negative, neutral) of the new textual data.

Use one of the following to create a demo interface for your client:

- Hugging Face Spaces using a Gradio App – [https://huggingface.co/spaces](https://huggingface.co/spaces)
- Streamlit Community Cloud (Streamlit Sharing) using a Streamlit App – [https://share.streamlit.io](https://share.streamlit.io)

---

## Dataset

Use the course evaluation dataset provided in class.

## Interpretation and Recommendation

Provide a brief interpretation of the results and a recommendation for the client.

- Interpret what the discovered topics mean and why certain sentiments dominate
- Provide recommendations based on your results. **Do not** recommend anything that is not supported by your results.

## Video Demonstration

Submit the link to a short video (not more than 4 minutes) demonstrating the topic modelling and the sentiment analysis.
Also include (in the same video) the user interface hosted on hugging face or streamlit.

| **Key**                             | **Value** |
|:------------------------------------|:----------|
| **Link to the video:**              |  https://drive.google.com/file/d/1k9U7ol7Tl0vG9wiHgFBlYws-Qb9-ar87/view?usp=sharing         |
| **Link to the hosted application:** |  https://natural-language-processing-202509-j-jhfmvxn25dg9hrsubrntr2.streamlit.app/         |

## Grading Approach

| Component                            | Weight | Description                                                       |
|:-------------------------------------|:-------|:------------------------------------------------------------------|
| **Data Preprocessing & Analysis**    | 20%    | Cleaning, preprocessing, and justification of chosen methods.     |
| **Topic Modelling**                  | 20%    | Correctness, interpretability, and coherence of topics.           |
| **Sentiment Analysis**               | 20%    | Appropriate model choice and quality of sentiment classification. |
| **Interface Design & Functionality** | 20%    | Usability, interactivity, and deployment success.                 |
| **Interpretation & Recommendation**  | 10%    | Logical, evidence-based, and actionable insights.                 |
| **Presentation (Video & Clarity)**   | 10%    | Clarity, professionalism, and demonstration of understanding.     |

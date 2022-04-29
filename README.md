# EECS 545 Final Project: Poetic Press
## Shrey Prakash Sahgal, Soo Hyun Ryu, Brandon Hardy, Chayce Baldwin
{shreyps,soohyunr,bkhardy,cbwin}@umich.edu

## Abstract
Can the news be poetic? In this project, we aim to use machine learning approaches to render the news in poetic style. 

Recent advances in natural language processing have effectively generated novel poetry with specific style and form. However, to date, models have not been developed to generate poetry while retaining the semantic content of input text. In this paper, we present the Poetic Press, which uses a state-of- the-art encoder-decoder model to summarize news articles and applies "poetic priors" to transfer news articles to the style of poetry while preserving semantic meaning. The "poetic priors" include a rhyme prior, which enforces a rhyme scheme, and word poeticness prior, which transforms the diction  to be more poetic and less dry. The results show that the abstractive model provided with prior  knowledge about poetry can render news articles in poetic style.


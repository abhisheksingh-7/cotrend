
<p align="center">
    <a href="#readme">
        <img alt="CoTrEnD Logo" src="static/cotrend.webp" style="height: 256px;">
    </a>
</p>
<h1 align="center" style="font-size: 2.5em; margin: 0; padding: 0;">Contrastively Trained Encodings from Decoder</h1>
<p align="center" style="font-size: 1.2em; font-weight: 300; color: #555; margin: 0;">
    Extending Decoders with an Integrated Encoder
</p>

This repo holds the code for training encoders that embed the final hidden state from large decoder models. To our knowledge, CoTrEnD is the first architecture to leverage a contrastive loss to train an encoder from a decoder. 

## Motivation
The motivation behind the CoTrEnD project is to utilize on the rich hidden states that are generated within large decoders. Rather than separating the embedder from the decoder as one typically would in a RAG approach, CoTrEnD integrates the encoder on top of the decoder. This allows the encoder to leverage the semantic information already captured within the decoder's hidden states.

## Architecture
The CoTrEnD architecture is a simple extension of the decoder-only model. The encoder is trained to embed the final hidden state of the decoder. The encoder is trained using a contrastive loss, which encourages the encoder to embed similar hidden states for similar inputs, and dissimilar hidden states for dissimilar inputs.

<p align="center">
    <a href="#readme">
        <img alt="CoTrEnD Logo" src="static/cotrend-architecture.png" style="height: 256px;">
    </a>
</p>
</p>


## Team
<div style="margin-bottom: 20px;">
    <h4 style="margin-bottom: 5px;">Abhishek Singh</h4>
    <div style="display: flex; align-items: center;">
        <a href="https://www.linkedin.com/in/abhisheksingh-7/" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin" alt="LinkedIn">
        </a>
        <a href="https://github.com/abhisheksingh-7" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github" alt="GitHub">
        </a>
        <a href="https://twitter.com/shekenotstirred" target="_blank">
            <img src="https://img.shields.io/twitter/follow/shekenotstirred?label=Follow&style=social" alt="Twitter">
        </a>
    </div>
</div>

<div style="margin-bottom: 20px;">
    <h4 style="margin-bottom: 5px;">Arthur Böök</h4>
    <div style="display: flex; align-items: center;">
        <a href="https://www.linkedin.com/in/arthurbook/" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin" alt="LinkedIn">
        </a>
        <a href="https://github.com/ArthurBook" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github" alt="GitHub">
        </a>
        <a href="https://twitter.com/TheRealABook" target="_blank">
            <img src="https://img.shields.io/twitter/follow/TheRealABook?label=Follow&style=social" alt="Twitter">
        </a>
    </div>
</div>

<div style="margin-bottom: 20px;">
    <h4 style="margin-bottom: 5px;">Wian Stipp</h4>
    <div style="display: flex; align-items: center;">
        <a href="https://www.linkedin.com/in/wian-stipp/" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin" alt="LinkedIn">
        </a>
        <a href="https://github.com/WianStipp" target="_blank" style="margin-right: 10px;">
            <img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github" alt="GitHub">
        </a>
        <a href="https://twitter.com/WianStipp" target="_blank">
            <img src="https://img.shields.io/twitter/follow/WianStipp?label=Follow&style=social" alt="Twitter">
        </a>
    </div>
</div>

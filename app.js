const boardInsights = {
  methods: {
    score: "84%",
    text: "Current synthesis emphasizes transparent scoring and source-linked comparisons across papers."
  },
  claims: {
    score: "79%",
    text: "Claim clustering makes repeated findings easier to compare without flattening important differences."
  },
  limits: {
    score: "71%",
    text: "The main risk is over-trusting generated summaries before checking the original source evidence."
  },
  gaps: {
    score: "68%",
    text: "Gap detection highlights missing datasets, weak comparisons, and questions that need follow-up reading."
  }
};

const demoData = {
  traceability: {
    tag: "Focus: Traceability",
    score: "0.84",
    title: "Source-linked synthesis",
    body: "The strongest artifact claim is that summaries become more trustworthy when every theme points back to the evidence cards that produced it.",
    method: "Evidence schema plus confidence scoring",
    limit: "Needs real corpus evaluation before research use",
    synthesis: "A responsible literature-review assistant should preserve the link between claims and sources. The prototype therefore treats synthesis as a map of evidence, not just a paragraph generator."
  },
  ranking: {
    tag: "Focus: Ranking",
    score: "0.78",
    title: "Relevance-first review planning",
    body: "Ranking helps a researcher decide which papers deserve close reading first, especially when the initial search returns too many loosely related results.",
    method: "Weighted relevance score across topic fit, method fit, and source confidence",
    limit: "Ranking criteria must remain visible so users can challenge the score",
    synthesis: "The ranking layer is useful when it remains explainable. Instead of hiding decisions behind a number, the artifact shows which evidence fields influenced priority."
  },
  gaps: {
    tag: "Focus: Gap Finding",
    score: "0.73",
    title: "Research gap surfacing",
    body: "Gap finding compares claims and limitations across evidence cards to expose missing baselines, weak datasets, and unanswered questions.",
    method: "Theme comparison across limitations and unresolved questions",
    limit: "Gap suggestions should be treated as prompts for reading, not final conclusions",
    synthesis: "The prototype frames research gaps as hypotheses. A gap is valuable only when a user can trace it to specific limitations or conflicts in the reviewed papers."
  }
};

document.querySelectorAll(".node").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".node").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");

    const topic = button.dataset.topic;
    document.querySelector("#confidenceScore").textContent = boardInsights[topic].score;
    document.querySelector("#insightText").textContent = boardInsights[topic].text;
  });
});

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");

    const data = demoData[button.dataset.filter];
    document.querySelector("#paperTag").textContent = data.tag;
    document.querySelector("#paperScore").textContent = data.score;
    document.querySelector("#paperTitle").textContent = data.title;
    document.querySelector("#paperBody").textContent = data.body;
    document.querySelector("#methodText").textContent = data.method;
    document.querySelector("#limitText").textContent = data.limit;
    document.querySelector("#synthesisText").textContent = data.synthesis;
  });
});

# Boston Public Schools Enrollment Chatbot — PRD

### TL;DR

A deeply conversational chatbot that guides parents and legal guardians in Boston through the BPS school enrollment journey. By asking targeted questions to build each user's unique profile—including address, grade, driving time, reputation and neighborhood preferences—the chatbot surfaces one clearly identified best recommendation with a brief rationale, plus three additional eligible schools presented in no particular order. Built in Python and deployed on HuggingFace, the system is engineered for accessibility, responsible recommendations, and user trust, always operating within BPS’s official rules.

---

## Goals

### Business Goals

* Successfully deploy a working chatbot on HuggingFace that accurately reflects current BPS eligibility rules as published on bostonpublicschools.org.

* Reduce the cognitive and time burden of understanding the BPS school enrollment landscape, especially for non-expert and non-English-first families.

* **Engineer a deeply conversational, question-driven experience:** the chatbot asks users all necessary questions before surfacing any results, building a comprehensive picture of user context—driving time, reputation, and neighborhood—tailored each session.

* **Generate a personalized preferences profile for each user** based on their answers including driving time, school reputation, and neighborhood fit.

* Clearly demonstrate responsible AI augmentation—chatbot provides recommendations, with rationale, while the human retains final decision authority.

* Produce a concise, high-quality evaluation memo summarizing methodology, performance, and results.

### User Goals

* Receive a single best school recommendation based on the user’s situation, accompanied by three additional eligible schools presented in no specific order. If the user asks for more, the chatbot returns the next three eligible schools in no specific order; this continues in batches of three until the pool of eligible schools matching the user's criteria is exhausted. The unordered presentation is intentional — to reduce anchoring bias and preserve the user's independent decision-making.

* Quickly determine which BPS schools their child is eligible to attend based on home address and grade.

* Filter eligible schools by attributes such as language programs, special education services, extracurriculars, and school hours.

* Visually compare schools via map and key feature attributes.

* Export or save the resulting school recommendations to share with family members.

* Resume a session later without re-entering all information, supporting interrupted or repeat interactions.


* er eligible schools by attributes such as language programs, special education services, extracurriculars, and school hours.

* Visually compare schools via map and key feature attributes.

* Export or save the resulting school recommendations to share with family members.

* Resume a session later without re-entering all information, supporting interrupted or repeat interactions.

### Non-Goals

* The chatbot will not make a school enrollment decision or recommendation on behalf of the user.

* The chatbot will not submit enrollment applications or interact with BPS internal/back-end systems.

* The system will not cover private schools or public schools outside of the BPS system.

* The chatbot will not recommend specific schools outside of BPS. If a user's situation (e.g., living near a state or city border) means no BPS school fits their needs well, the chatbot may explain why BPS may not be the right fit and suggest the user explore other area schools—without recommending specific options outside BPS.

---

## User Stories

**Persona:** Parent / Legal Guardian

* As a parent, I want to enter my home address and my child’s grade level so that I can see which BPS schools my child is eligible to attend.

* As a parent, I want to filter schools by language programs (e.g., Spanish bilingual) so my child can learn in our home language.

* As a parent, I want to see each eligible school’s distance from my home or workplace and nearby public transit options, so I can consider commute constraints.

* As a parent, I want to compare schools side-by-side on grade levels, programs, extracurriculars, hours, and other key attributes.

* As a parent, I want to save or download my recommended school list to share with other family members before making an enrollment decision.

* As a parent with a child who has special needs, I want to filter schools by the special education services offered.

* As a parent with limited time, I want a quick list of eligible schools with minimal required input.

* **As a parent who feels lost and overwhelmed, I want someone to guide me from scratch so that I understand what questions I should even be asking about school enrollment.**

---

## Functional Requirements

* **Conversational Intake (Priority: P0)**

  * Address Collection: Gather user location by **ZIP code alone, if the eligibility API permits**; request full street address, apt/unit, city, state, and zip ONLY if required for eligibility lookup.

  * Grade Level Capture: Intake child’s current or next grade (e.g., K1, K2, 3rd grade, etc.).

  * Home Language/Language Program Option: Allow user to specify language program needs or preferences.

  * Optional: Capture workplace address for commute-based sorting/ranking.

* **Eligibility Engine (Priority: P0)**

  * Map provided address (ZIP or full address as available) and grade to eligible BPS schools using the latest assignment rules.

  * Only present schools for which the child is eligible, and clearly indicate zoning/eligibility logic.

  * **Eligibility-Based Guardrails:** The chatbot enforces conversation breaks for:

    * *Age/Grade Check*: If the user enters an age or grade not corresponding to a school-age child (e.g., their grandparent), the chatbot gently redirects and ends intake.

    * *Geography Check*: If the user's ZIP or address is outside Boston/Massachusetts, the chatbot informs the user that BPS only serves Boston families and exits intake flow.

* **Preference Filtering & Ranking (Priority: P1)**

  * Filter results by user-stated preferences (language program, special education, extracurriculars, school hours).

  * **Build a personalized user profile**—including driving time, school reputation, and neighborhood fit—based on the conversation.

  * Rank results by match to preferences and, where relevant, by distance.

* School Output & Comparison (Priority: P1) — The chatbot identifies and presents one best recommendation with a brief rationale. Alongside it, three additional eligible schools are displayed in no specific order. If the user requests more, the next three are returned in no specific order; this continues in batches of three until the eligible pool is exhausted. Schools beyond the single best recommendation are intentionally unordered to minimize ranking bias and support independent user decision-making.

* endation are intentionally unordered to minimize ranking bias and support independent user decision-making.

* The chatbot identifies and presents one best recommendation with a brief rationale. Alongside it, three additional eligible schools are displayed in no specific order. If the user requests more, the next three are returned in no specific order; this continues in batches of three until the eligible pool is exhausted. Schools beyond the single best recommendation are intentionally unordered to minimize ranking bias and support independent user decision-making.

* 

  * owser) and return without re-entering prior information.

  * Enable export/download of the recommended school list, either as PDF or CSV.

* **Accuracy & Guardrails (Priority: P0)**

  * Ensure the chatbot always bases responses on authoritative BPS information—never fabricates eligibility data.

  * Clearly cite sources and link out to BPS enrollment information for verification.

  * Display persistent disclaimers emphasizing that users should confirm eligibility with BPS before acting.

---

## User Experience

**Entry Point & First-Time User Experience**

* **Users access the chatbot ONLY through HuggingFace Spaces; remove all other entry points.**

* No account creation or login required for standard use.

* The application must be **runnable and testable locally** (support local execution in app.py as part of basic developer workflow).

* On entry, users are greeted by a warm, plain-language message:

  * This message outlines that the chatbot will guide them step-by-step through their school exploration, cannot enroll them, and that all recommendations are for informational purposes.

* Optional onboarding tutorial is available but not required.

**Core Experience**

* The interaction is entirely question-based and conversational: The chatbot asks for one piece of information at a time, starting from scratch and guiding the user. Examples: "Let’s start! What grade will your child be entering next year?" "What ZIP code do you and your child live in?" "Do you have any specific needs or preferences for your child’s school (like a language program, special education services, after-school care, or location)?" No options or preference lists are presented upfront; context is earned through dialogue.

* Eligibility Checkpoints: After grade is entered, the chatbot checks if it matches a school-age child; if not, it gently exits or redirects. After address/ZIP is entered, checks if address is within Boston; if not, informs user BPS serves Boston residents and stops intake. At each step, the chatbot asks clarifying questions until it has all the context needed to generate a tailored recommendation.

* School Recommendation Output: The chatbot presents one highlighted best recommendation with a short rationale, plus three additional schools shown in no particular order. If the user asks for more, the chatbot returns the next three eligible schools in no specific order, continuing until the pool is exhausted. The unordered display is a deliberate design choice to reduce anchoring bias.

* Follow-Up & Comparison: User can ask further questions ("Which have Spanish bilingual programs?" "What if I work on Blue Hill Ave?"). User can select up to three schools for side-by-side comparison of features. User can export or download recommendations for later review. The chatbot supports session saving and recovery for interrupted use.


* ser can export or download recommendations for later review.The chatbot supports session saving and recovery for interrupted use.

**Advanced Features & Edge Cases**

* If address/ZIP is outside Boston, explain BPS serves Boston families only and stop further intake.

* If user attempts to enroll someone not school-aged, explain eligibility boundaries.

* If eligibility is ambiguous (e.g., address on district boundary), flag issue and recommend direct follow-up with BPS.

* Visible disclaimers throughout:  

  *“This chatbot helps you explore your options. Always confirm eligibility with Boston Public Schools directly.”*

**UI/UX Highlights**

* Responsive design with mobile-first approach.

* Accessible, jargon-free language; multilingual access (Spanish, Haitian Creole, Chinese, Vietnamese prioritized).

* Clean map integration visible on all devices.

* Accessibility compliance for contrast, keyboard navigation, and ARIA labels.

* Minimal bandwidth and smartphone-friendly for maximum reach.

---

## Narrative

Maria is a 34-year-old mother living in Boston, working part-time and commuting most days by bus. Her son, Javier, just completed second grade, and Maria knows it’s time to find him a good placement for third grade in Boston Public Schools. While Maria is committed to her son’s success, she’s not fluent in English and finds the BPS website confusing and overwhelming. Time is short—she only has a few minutes during her commute.

One afternoon, Maria discovers the BPS Enrollment Chatbot on her phone. She’s greeted warmly, and the chatbot guides her step-by-step, asking specific, bite-sized questions about grade, address (ZIP code is accepted), and her unique needs. When prompted, Maria mentions a preference for a Spanish bilingual program and her need for after-school care. After confirming that her address is within Boston, the chatbot quickly surfaces a best-fit recommendation and two more strong options, clearly marked with program details, neighborhood, and commute info. A simple map also displays school locations and transit lines.

Maria asks, “Which school is easiest for the 22 bus?” and the chatbot highlights relevant options. She exports the summary and shares it with her husband. Prepared, Maria visits the BPS Welcome Center already focused on her top choices—her preferences and context honored, her decision process streamlined. For BPS, this reduces confusion, saves staff time, and increases parent satisfaction and trust.

---

## Success Metrics

### User-Centric Metrics

* Percentage of sessions resulting in a school recommendation (>70% target).

* Average user satisfaction rating at the end of the session (target >4/5).

* Percentage of users who complete the address/ZIP intake process.

* Frequency of session recovery (users resuming after interruption).

### Business Metrics

* Number of unique chatbot sessions on HuggingFace.

* Accuracy of eligibility recommendations vs. verified BPS ground truth (target >90%).

* Proportion of BPS eligibility rules represented.

### Technical Metrics

* Average response latency per chatbot turn (<3 seconds).

* Rate of eligibility-related errors or chat “hallucinations” (<5%).

* Uptime/reliability on HuggingFace’s free hosting tier.

### Tracking Plan

* Session start event

* Grade level entered event

* Address/ZIP submitted event

* Preference profile completed event

* School recommendation displayed event

* Follow-up question/clarification asked event

* Export/download triggered event

* Session abandoned (last completed step)

---

## Technical Considerations

### Technical Needs

* Python-based application (single app.py or equivalent).

* LLM backbone (via HuggingFace/Instruct pipeline) for conversational intake.

* Deterministic eligibility logic for address/ZIP + grade → school mapping, referencing official BPS boundaries.

* Structured school data (attributes, programs, services).

* Map rendering via Google Maps or OpenStreetMap API.

* Session-based storage for continuity.

### Integration Points

* Data import and regular updates from bostonpublicschools.org and boston.explore.avela.org.

* Hosted on HuggingFace Spaces for public access and local testing.

* Optional: multilingual translation APIs.

### Data Storage & Privacy

* **No persistent storage of user addresses or PII outside active session.**

* Sessions saved only in-browser/local; no server-side PII storage.

* Clear privacy notice at entry.

* Special consideration for undocumented or at-risk families—no unnecessary data retention.

### Scalability & Performance

* Designed for HuggingFace’s free tier (moderate compute/serverless).

* Deterministic/explicit eligibility logic for all eligibility checks to reduce LLM hallucination.

* Handles modest simultaneous load efficiently.

### Potential Challenges

* BPS rules are complex and updated annually; the eligibility engine must be easy for non-developers to update.

* Zero tolerance for eligibility “hallucinations”; all fact claims must be source-cited.

* Multilingual and accessibility requirements are mandatory for trust and equity.

---

## Milestones & Sequencing

### Project Estimate

* Medium project: 2–4 weeks to MVP.

### Team Size & Composition

* Small Team (2–3 people)

  * 1 Product/Design Lead for requirements, UX, and documentation.

  * 1–2 Engineers for backend, data assembly, conversation logic, and deployment.

### Suggested Phases

**Phase 1 — Foundation (Week 1)**

* Parse and structure official BPS data; set up HuggingFace Space and Python scaffold; build and test eligibility engine (address/ZIP + grade).

**Phase 2 — Conversational Interface (Week 2)**

* Integrate LLM for Q&A, implement guided intake flow, connect to eligibility logic.

**Phase 3 — Enrichment & Polish (Week 3)**

* Complete preference/profile-based filtering and ranking, school comparison, map view, export. Begin Spanish language support.

**Phase 4 — Evaluation (Week 4)**

* Conduct real-world accuracy and usability tests; capture defined metrics; deliver final summary memo.

---
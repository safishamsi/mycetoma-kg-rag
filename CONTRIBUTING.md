# Contributing to Mycetoma KG-RAG

Thank you for your interest in contributing to the Mycetoma Knowledge Graph-Augmented Retrieval system! This project aims to improve mycetoma diagnosis globally through community-driven knowledge sharing.

## 🌍 Our Mission

Build the world's most comprehensive, open-access knowledge base for mycetoma diagnosis to:
- Improve diagnostic accuracy in under-resourced areas
- Enable data-driven treatment decisions
- Accelerate research on neglected tropical diseases
- Democratize access to AI-powered medical diagnostics

## 🤝 How to Contribute

We welcome contributions in multiple forms:

### 1. 📊 Data Contributions

#### **Clinical Cases**
Share anonymized patient cases with histopathology images to grow our Knowledge Graph.

**Requirements:**
- ✅ Patient consent obtained
- ✅ IRB/Ethics committee approval
- ✅ Data fully anonymized (no patient identifiers)
- ✅ Histopathology images (H&E stained preferred)
- ✅ Clinical notes and demographics
- ✅ Laboratory confirmation (if available)

**How to Submit:**
1. Fill out the [Case Contribution Template](data/contribution/CONTRIBUTION_TEMPLATE.yaml)
2. Upload via our [Data Portal](https://mycetoma-kg.org/contribute) (coming soon)
3. Or email to: contributions@mycetoma-kg.org
4. Your contribution will be reviewed within 2 weeks

**What You Get:**
- 🏆 Recognition in CONTRIBUTORS.md
- 📜 Co-authorship on follow-up papers (for major contributions)
- 📊 DOI for your contributed dataset
- 🌟 Attribution in all uses of your data

#### **Literature References**
Help expand our biomedical literature database.

**How to Contribute:**
- Submit relevant PubMed IDs (PMIDs)
- Provide full-text PDFs for non-indexed papers
- Annotate papers with pathogen/location/drug information

### 2. 💻 Code Contributions

#### **Areas Needing Help:**
- 🔧 Bug fixes
- ⚡ Performance optimization
- 🧪 New diagnostic models
- 🌐 Multi-language support
- 📱 Mobile app development
- 🔬 New retrieval modalities

#### **Development Process:**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with clear commit messages
4. **Add** tests for new functionality
5. **Ensure** all tests pass (`pytest tests/`)
6. **Submit** a Pull Request

**Code Standards:**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests (target: 80% coverage)
- Update documentation as needed

### 3. 📚 Documentation Contributions

Help make our project accessible to researchers worldwide:
- 📖 Improve setup guides
- 🎓 Create tutorials
- 🌐 Translate documentation
- 📹 Record video tutorials
- 📝 Write blog posts about use cases

### 4. 🔬 Expert Review

Pathologists and mycetoma experts can help by:
- Validating contributed cases
- Reviewing diagnostic predictions
- Providing clinical feedback
- Testing the system in clinical settings

**Apply for Expert Reviewer status:** experts@mycetoma-kg.org

### 5. 🐛 Bug Reports & Feature Requests

Found an issue or have an idea?
- [Report bugs](https://github.com/yourusername/mycetoma-kg-rag/issues/new?template=bug_report.md)
- [Request features](https://github.com/yourusername/mycetoma-kg-rag/issues/new?template=feature_request.md)
- Participate in [Discussions](https://github.com/yourusername/mycetoma-kg-rag/discussions)

---

## 📋 Contribution Guidelines

### Data Quality Standards

All contributed cases must meet these criteria:

#### **Required Information:**
- Patient demographics (age, gender, location)
- Anatomical site affected
- Disease duration
- Clinical presentation
- Diagnosis (Actinomycetoma vs Eumycetoma)
- At least one histopathology image (H&E stained)

#### **Encouraged Information:**
- Laboratory confirmation (culture, PCR, histopathology)
- Confirmed causative organism
- Treatment history
- Outcome data
- Multiple imaging modalities
- Grain characteristics

#### **Image Quality:**
- Minimum resolution: 1024x1024 pixels
- Format: JPEG or PNG
- Color: 24-bit RGB
- Proper focus and lighting
- Representative grain morphology visible

### Ethical Requirements

**CRITICAL**: All data contributions MUST comply with:

1. **Patient Consent**
   - Informed consent for research use
   - Consent for data sharing (even if anonymized)
   - Document consent in contribution form

2. **IRB/Ethics Approval**
   - Provide IRB approval number
   - Upload approval letter (redacted if needed)
   - Confirm study follows Helsinki Declaration

3. **Anonymization**
   - Remove all patient identifiers
   - No faces in images
   - No metadata with location/hospital details
   - Generalize dates (year only)

4. **Data Sharing Agreement**
   - Accept CC BY-NC-SA 4.0 license
   - Understand data will be publicly available
   - Agree to attribution requirements

**We take data privacy seriously.** Violations will result in immediate removal from the project.

### Code Review Process

All Pull Requests must:
- ✅ Pass automated tests
- ✅ Have at least one approving review
- ✅ Follow our coding standards
- ✅ Include documentation updates
- ✅ Not break existing functionality

**Review Timeline:** We aim to review PRs within 7 days.

---

## 🏆 Recognition & Authorship

### Contributor Levels:

#### **🥇 Major Contributors**
Contributions: 50+ cases OR significant code features
Recognition:
- Listed in CONTRIBUTORS.md (top section)
- Co-authorship on methodology papers
- Invited to steering committee
- Logo on website (institutions)

#### **🥈 Active Contributors**
Contributions: 10-49 cases OR multiple code contributions
Recognition:
- Listed in CONTRIBUTORS.md
- Acknowledged in papers
- Early access to new features

#### **🥉 Contributors**
Contributions: 1-9 cases OR minor code contributions
Recognition:
- Listed in CONTRIBUTORS.md
- Citation in dataset DOI

### Publication Policy

**Primary Paper:**
- Core development team only

**Follow-up Papers:**
- Major contributors offered co-authorship
- Contributors acknowledged in acknowledgments
- Institutional contributors thanked

**Your Own Papers:**
- ✅ Use our KG and cite appropriately
- ✅ Publish your analyses
- ✅ We encourage derivative work!
- 📧 Let us know: We'll feature your work on our website

---

## 🔒 Security & Privacy

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
- 📧 Email: security@mycetoma-kg.org
- 🔐 Use PGP key: [link to public key]
- ⏱️ We'll respond within 48 hours

### Data Security

When contributing:
- 🚫 Never upload identifiable patient data
- 🚫 Never share credentials or API keys
- ✅ Use HTTPS for all transfers
- ✅ Encrypt sensitive information

---

## 💬 Communication Channels

### For Contributors:
- **GitHub Discussions:** Technical questions, feature ideas
- **Discord:** Real-time chat (invite: discord.gg/mycetoma)
- **Mailing List:** Monthly updates (subscribe: newsletter@mycetoma-kg.org)
- **Twitter:** @MycetomeKG - Latest news

### Monthly Community Calls:
- **When:** First Tuesday of each month, 14:00 UTC
- **Where:** Zoom (link in Discord)
- **Topics:** Roadmap, new features, Q&A

---

## 📅 Roadmap & Priorities

### Current Focus (Q4 2025):
- ✅ Onboard first 10 contributing institutions
- ✅ Reach 1,000 total cases
- ✅ Add Spanish and Hindi languages
- ✅ Launch mobile app beta

### Next Quarter (Q1 2026):
- ✅ Federated learning deployment
- ✅ Integration with OpenMRS
- ✅ API v2.0 with advanced queries

See our full [Roadmap](docs/ROADMAP.md)

---

## 📖 Additional Resources

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License Information](LICENSE.md)
- [Ethical Guidelines](docs/ETHICAL_GUIDELINES.md)
- [API Documentation](docs/api_reference.md)
- [FAQ](docs/FAQ.md)

---

## 🙏 Acknowledgments

This project is made possible by:
- **Mycetoma Research Centre**, University of Khartoum, Sudan
- **WHO Collaborating Centre on Mycetoma**
- All our amazing contributors (see [CONTRIBUTORS.md](CONTRIBUTORS.md))
- Funding from [your grants]

---

## 📬 Contact

- **General inquiries:** info@mycetoma-kg.org
- **Data contributions:** contributions@mycetoma-kg.org
- **Technical support:** support@mycetoma-kg.org
- **Partnerships:** partnerships@mycetoma-kg.org

---

## ⚖️ License

By contributing, you agree that your contributions will be licensed under:
- **Code:** MIT License
- **Data:** CC BY-NC-SA 4.0 License

See [LICENSE](LICENSE) for full terms.

---

**Thank you for helping us fight mycetoma globally! 🌍❤️**

Every contribution, no matter how small, makes a difference in improving patient outcomes.

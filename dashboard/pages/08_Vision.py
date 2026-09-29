import streamlit as st

st.title("🔭 Vision & Roadmap")
st.caption("Where ADIS goes after the MVP.")

st.subheader("Near-term")
cols = st.columns(3)
cols[0].markdown("### Investigation\nEvidence enrichment, behavior profiles, ATT&CK mapping, context.")
cols[1].markdown("### Adaptive Response\nPolicy-driven isolation, validation, recovery and safer automation.")
cols[2].markdown("### Evaluation\nContinual regression, adversarial testing and first-vs-subsequent exposure measurement.")

st.write("")
st.subheader("Long-term architecture")
st.markdown("""
```text
                    ┌─────────────────────────┐
                    │   Continuous Telemetry  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Detection / Self Model  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Behavioral Understanding│
                    └────────────┬────────────┘
                                 ↓
              ┌──────────────────┴──────────────────┐
              ↓                                     ↓
       Persistent Memory                      Investigation
              │                                     │
              └──────────────────┬──────────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Risk + Policy + Safety  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Containment / Recovery  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Validate → Learn → Adapt │
                    └─────────────────────────┘
```
""")

st.subheader("Future extensions")
for x in [
    "Continual learning with regression protection against catastrophic forgetting.",
    "Automated adversarial evaluation / controlled red-team generation.",
    "Richer deception environments and deeper attacker behavior collection.",
    "Threat-intelligence enrichment and MITRE ATT&CK mapping.",
    "Evidence-grounded RAG investigation assistant.",
    "Shared cyber + physical immune architecture as a future research direction.",
]:
    st.write("→ " + x)

st.warning("These are roadmap directions, not claims that every capability is already implemented in the MVP.")

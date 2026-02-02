import streamlit as st
import pulp
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- AI LAYER ---
class ProjectNeeds(BaseModel):
    required_skill: str = Field(description="The technical skill required")
    estimated_hours: int = Field(description="The hours the task will take")

def run_ai_logic(api_key, query, model_name):
    """
    Utilizes Generative AI to parse unstructured project requests 
    into structured data for optimization.
    """
    llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
    parser = JsonOutputParser(pydantic_object=ProjectNeeds)
    prompt = ChatPromptTemplate.from_template(
        "Analyze this project request: {query}\n\n{format_instructions}"
    )
    chain = prompt | llm | parser
    return chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})

# --- ETHICAL AI & TRANSPARENCY LAYER ---
def explain_decision(result, skill, hours, df):
    """
    Provides a transparency report to explain the 'Why' behind the 
    mathematical optimization, ensuring accountability.
    """
    if not result:
        return "No assignment made. Constraints (Availability/Skill) were too strict."
    
    selected_name = result['Consultant']
    current_rate = df[df['Consultant'] == selected_name]['Hourly_Rate'].values[0]
    
    # Check for candidates who were cheaper but lacked availability
    cheaper_candidates = df[
        (df['Skill'].str.contains(skill, case=False, na=False)) & 
        (df['Hourly_Rate'] < current_rate)
    ]
    
    explanation = f"**Optimization Audit Trail:**\n"
    explanation += f"- **Selected:** {selected_name} provides the lowest cost while meeting the {hours}-hour availability threshold.\n"
    
    if not cheaper_candidates.empty:
        unavailable_count = len(cheaper_candidates)
        explanation += f"- **Fairness Check:** {unavailable_count} consultant(s) had lower rates but were excluded due to insufficient availability hours.\n"
    else:
        explanation += "- **Fairness Check:** No other qualified candidates with a lower hourly rate were found."
        
    return explanation

# --- OPTIMIZATION LAYER ---
def solve_mip(df, skill, hours):
    """
    Mixed Integer Programming (MIP) to find the globally optimal 
    resource allocation based on cost and constraints.
    """
    mask = (df['Skill'].str.contains(skill, case=False, na=False)) & (df['Availability_Hours'] >= hours)
    qualified = df[mask].copy()
    
    if qualified.empty:
        return None

    # Objective: Minimize total project cost
    prob = pulp.LpProblem("Resource_Optimization", pulp.LpMinimize)
    names = qualified['Consultant'].tolist()
    x = pulp.LpVariable.dicts("assign", names, cat=pulp.LpBinary)
    
    prob += pulp.lpSum([
        x[n] * qualified.loc[qualified['Consultant'] == n, 'Hourly_Rate'].values[0] * hours 
        for n in names
    ])
    
    # Constraint: Exactly one consultant must be assigned
    prob += pulp.lpSum([x[n] for n in names]) == 1
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    for n in names:
        if pulp.value(x[n]) == 1:
            return {"Consultant": n, "Cost": pulp.value(prob.objective)}
    return None

# --- UI LAYER ---
st.set_page_config(page_title="OptiTask AI Enterprise", layout="wide")
st.title("💼 OptiTask AI")

# Initialize Session States
if "needs" not in st.session_state:
    st.session_state.needs = {"required_skill": "", "estimated_hours": 0}
if "resource_df" not in st.session_state:
    st.session_state.resource_df = None

with st.sidebar:
    st.header("Control Panel")
    user_api_key = st.text_input("Enter Groq API Key", type="password")
    selected_model = st.selectbox("Select Reasoning Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    
    st.divider()
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Resource Master (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.resource_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(st.session_state.resource_df)} records.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Requirement Extraction")
    client_query = st.text_area("Describe the project needs:", "We need an Azure Cloud expert for 25 hours.")
    
    if st.button("Extract via AI"):
        if not user_api_key:
            st.error("Please provide an API Key to proceed.")
        else:
            with st.spinner("AI parsing project parameters..."):
                try:
                    st.session_state.needs = run_ai_logic(user_api_key, client_query, selected_model)
                except Exception as e:
                    st.error(f"AI Extraction Failed: {e}")

    st.divider()
    st.subheader("2. Human-in-the-Loop Validation")
    # Allows manual overrides for AI findings to ensure accuracy
    edit_skill = st.text_input("Verified Skill:", value=st.session_state.needs.get("required_skill", ""))
    edit_hours = st.number_input("Verified Hours:", value=int(st.session_state.needs.get("estimated_hours", 0)))

    if st.button("Generate Optimal Assignment"):
        if st.session_state.resource_df is not None:
            with st.spinner("Running MIP Solver..."):
                result = solve_mip(st.session_state.resource_df, edit_skill, edit_hours)
                if result:
                    st.success(f"✅ Optimal Allocation: {result['Consultant']}")
                    st.metric("Minimum Project Cost", f"${result['Cost']:.2f}")
                    
                    # Trusted AI Panel
                    with st.expander("🛡️ Transparency & Ethical AI Report"):
                        st.markdown(explain_decision(result, edit_skill, edit_hours, st.session_state.resource_df))
                        st.info("Accountability: Selection is based on objective data constraints to mitigate unconscious bias.")
                else:
                    st.error("Optimization Failed: No available resources meet the combined skill and hour constraints.")
        else:
            st.warning("Please upload the resource dataset in the sidebar.")

with col2:
    st.subheader("Resource Inventory Preview")
    if st.session_state.resource_df is not None:
        st.dataframe(st.session_state.resource_df, height=600, use_container_width=True)
    else:
        st.info("Awaiting CSV upload for data analysis.")
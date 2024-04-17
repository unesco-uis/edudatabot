import streamlit as st
from lida import Manager, llm
from llmx import  TextGenerationConfig
from lida.datamodel import Goal, Summary, Persona
import os
import pandas as pd
import base64  

text_gen = llm(
    provider="openai",
    api_type="azure",
    azure_endpoint=os.environ["AZURE_OPENAI_BASE"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_VERSION"],
)

#export AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo


# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="EduDataBot: Automatic Generation of Visualizations and Infographics - Demo",
    page_icon="static/unesco-16-168843.png"
)

# Function to get base64 string


def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Convert your image to base64 string
logo_base64 = get_image_base64("static/UNESCO_UIS_logo_color_eng.jpg")

# Embed the base64 string directly in the HTML
st.sidebar.markdown(
    f"""
    <div style='text-align: left; padding: 0 0 1em 0;'>
        <img src="data:image/jpeg;base64,{logo_base64}" style='max-width: 150px; height: auto;'>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("# EduDataBot: An AI-enhanced data visualization tool using SDG 4 data")

#st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key
openai_key = os.getenv("AZURE_OPENAI_API_KEY")


#if not openai_key:
#    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
#    if openai_key:
#        display_key = openai_key[:2] + "*" * \
#            (len(openai_key) - 5) + openai_key[-2:]
#        st.sidebar.write(f"Current key: {display_key}")
#    else:
#        st.sidebar.write("Please enter OpenAI API key.")
#else:
#    display_key = openai_key[:2] + "*" * \
#        (len(openai_key) - 5) + openai_key[-3:]
#    st.sidebar.write(
#        f"Azure OpenAI API key loaded from environment variable: {display_key}")

st.markdown(
    """
    This prototype version of the EduDataBot is designed to harness the power of machine learning to better visualize education data. The next development of the tool will include an interactive interface powered by an advanced chatbot. 
    Fed by the reliable database from the UNESCO Institute for Statistics for SDG 4, it provides a unique and valuable resource for informed decision-making.

    EduDataBot leverages python libraries like Llamaindex, Lida, Trulens and AffectLog for generating data visualizations and data-faithful infographics and is grammar agnostic (will work with any programming language and visualization
    libraries e.g. matplotlib, seaborn, altair, d3 etc) and works with multiple large language model providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface). 
    See the project page [here](https://github.com/unesco-uis/un-vision-ai) for updates.

""")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Summary","Goals","Visualisation", "Explanation", "Evaluation", "Viz Code"])

# Step 2 - Select a dataset and summarization method
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    selected_model = "gpt-35-turbo"

    # select temperature on a scale of 0.0 to 1.0
    temperature = 1

    # set use_cache in sidebar
    use_cache = True
    #use_cache = st.sidebar.checkbox("Use cache", value=True)


    # Handle dataset selection and upload

    st.sidebar.write("### Please choose a dataset below to start")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Education facilities and safety",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Education_facilities_and_safety.csv"},
        {"label": "Equal access",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Equal_access.csv"},
        {"label": "Literacy and numeracy",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Literacy_and_numeracy.csv"},
        {"label": "Pre-primary education",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Pre_primary_education.csv"},
        {"label": "Primary and secondary education",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Primary_and_secondary_education.csv"},
        {"label": "Scholarships",
            "url": "/app/datasets/Scholarships.csv"},
        {"label": "Skills", "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Skills.csv"},
        {"label": "Sustainable development knowledge",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Sustainable_development_knowledge.csv"},
        {"label": "Teachers", "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Teachers.csv"},
        {"label": "Technical vocational and tertiary",
            "url": "https://raw.githubusercontent.com/unesco-uis/edudatabot/main/datasets/Technical_vocational_and_tertiary.csv"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Available SDG4 datasets',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or JSON file", type=["csv", "json"])

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            #st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"] for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info(
            "To continue, select a dataset from the sidebar on the left.")

    # st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    #summarization_methods = [
    #    {"label": "llm",
    #     "description":
    #     "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
    #    {"label": "default",
    #     "description": "Uses dataset column statistics and column names as the summary"},
    #    {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    #selected_method_label = st.sidebar.selectbox(
    #    'Choose a method',
    #    options=[method["label"] for method in summarization_methods],
    #    index=0
    #)
    selected_method = "default" 
    #selected_method = summarization_methods[[
    #    method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = "Using dataset column statistics and column names as the summary"
    #selected_summary_method_description = summarization_methods[[
    #    method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)
            
    
    #st.sidebar.write("### Who are you ?")
    #persona = st.sidebar.text_input("Tell us who you are (e.g. data expert, education expert, policy maker", value = "data expert")


# Step 3 - Generate data summary
    #tab1.write("## Data Summarization")
if openai_key and selected_dataset and selected_method:
    #lida = Manager(text_gen=llm("openai", api_key=openai_key))
    lida = Manager(text_gen=text_gen)
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    tab1.write("## Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        tab1.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        tab1.write(nfields_df)
    else:
        tab1.write(str(summary))

    # Step 4 - Generate goals
    if summary:
        tab2.write("### Goal Selection")

        num_goals = tab2.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = tab2.checkbox("Add Your Own Goal")

        # **** lida.goals *****
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        tab2.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = tab2.text_input("Describe Your Goal")

            if user_goal:

                new_goal = Goal(question=user_goal,
                                visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = tab2.selectbox(
            'Choose a generated goal', options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        #st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

        # Step 5 - Generate visualizations
        if selected_goal_object:
            #st.sidebar.write("## Visualization Library")
            #visualization_libraries = ["seaborn", "matplotlib", "plotly", "ggplot"]
            selected_library = "seaborn"
            #selected_library = st.sidebar.selectbox(
            #    'Choose a visualization library',
            #    options=visualization_libraries,
            #    index=0
            #)

            # Update the visualization generation call to use the selected library.
            tab3.write("## Visualizations")

            # slider for number of visualizations
            num_visualizations = tab3.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=5,
                value=3)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, 
                temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            viz_titles = [
                f'Visualization {i+1}' for i in range(len(visualizations))]

            selected_viz_title = tab3.selectbox(
                'Choose a visualization', options=viz_titles, index=0)

            selected_viz = visualizations[viz_titles.index(selected_viz_title)]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                tab3.image(img, caption=selected_viz_title,
                         use_column_width=True)
            
            

            # **** lida.visualize *****
            instructions = tab3.text_input("Add instructions to refine the visualisation")
            
            if instructions:
                
                code = selected_viz.code
            
                edited_charts = lida.edit(code=code,
                    summary=summary, 
                    instructions=instructions, 
                    library=selected_library, 
                    textgen_config=textgen_config)
                    
                edited_viz = edited_charts[viz_titles.index(selected_viz_title)]
                
                imgdata = base64.b64decode(edited_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                tab3.image(img, caption=selected_viz_title,
                         use_column_width=True)
                    
        # Step 4 - Generate visualizations
            if instructions:
              
              explanations = lida.explain(code=edited_viz.code, library=selected_library, textgen_config=textgen_config)
          
              for row in explanations[viz_titles.index(selected_viz_title)]: 
                tab4.write(row["section"] + ": " + row["explanation"])
            
            else:
              
              explanations = lida.explain(code=selected_viz.code, library=selected_library, textgen_config=textgen_config)
          
              for row in explanations[viz_titles.index(selected_viz_title)]: 
                tab4.write(row["section"] + ": " + row["explanation"])
            
        
         # Step 5 - Evaluation
            if instructions:
              
              evaluations = lida.evaluate(code=edited_viz.code,  goal=goals[viz_titles.index(selected_viz_title)], textgen_config=textgen_config, library=selected_library)[0]
              
              for eval in evaluations:
                tab5.write(eval["dimension"])
                tab5.write(eval["score"])
                tab5.write(eval["rationale"][:200])
                
            else:
              
              evaluations = lida.evaluate(code=selected_viz.code,  goal=goals[viz_titles.index(selected_viz_title)], textgen_config=textgen_config, library=selected_library)[0]
              
              for eval in evaluations:
                tab5.write(eval["dimension"])
                tab5.write(eval["score"])
                tab5.write(eval["rationale"][:200])
                
         # Step 6 - Code
         
            #tab6.write("### Visualization Code")
            tab6.code(selected_viz.code)
            
            
         
         

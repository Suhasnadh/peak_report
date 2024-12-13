import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config with dark theme
st.set_page_config(
    page_title="Housing Analysis",
    layout="wide",
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #0e1117;
    }
    .css-1r6slb0 {color: #ffffff !important;}
    .css-1wivap2 {color: #ffffff99 !important;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
    }
    .stSelectbox [data-baseweb="select"] {
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def get_sorted_unique_values(df, column):
    """Get sorted unique values from a column, handling mixed types."""
    values = df[column].unique()
    values = [str(x) for x in values if pd.notna(x)]
    return ['All'] + sorted(values)

def preprocess_data(df):
    """Clean and preprocess the data."""
    df = df.copy()
    df['Membership Status'] = df['Membership Status'].fillna('')
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['Student Number'], keep='first')
    rows_removed = initial_rows - len(df)
    return df, rows_removed

def standardize_community_name(name):
    """Standardize community names."""
    if pd.isna(name):
        return name
        
    name = str(name).strip().upper()
    
    community_mapping = {
        'PRIDE': ['PRIDE', 'PRIDE COMMUNITY', 'PRIDE HOUSING'],
        'BLACK AFFINITY': ['BLACK AFFINITY', 'BLACK AFFINITY COMMUNITY', 'BAH'],
        'VIKING LAUNCH': ['VIKING LAUNCH', 'VIKING LAUNCH COMMUNITY'],
        'LA COMUNIDAD': ['LA COMUNIDAD', 'LACO', 'LA CO'],
        'HOUSING AMBASSADOR': ['HOUSING AMBASSADOR', 'HA', 'HOUSING AMBASSADORS']
    }
    
    for standard_name, variations in community_mapping.items():
        if name in variations or any(var in name for p in variations if p in name):
            return standard_name
            
    return name

def analyze_compatibility(str1, str2):
    """Analyze compatibility between two roommate questionnaire strings."""
    try:
        if not str1 or not str2 or pd.isna(str1) or pd.isna(str2):
            return None
        
        str1 = str(str1).strip()
        str2 = str(str2).strip()
        
        if len(str1) != 5 or len(str2) != 5:
            return None
        
        matches = sum(1 for i in range(4) if str1[i] == str2[i])
        preference_match = str1[-1] == str2[-1]
        
        return matches + (1 if preference_match else 0)
        
    except (TypeError, IndexError):
        return None

def get_community_stats(df):
    """Get standardized community statistics."""
    community_stats = []
    
    # Handle Honors
    honors_interested = len(df[df['Community Interest'] == 'Honors Community'])
    honors_active = len(df[(df['Honors'] == 'Honors') & (df['Membership Status'] == 'Active')])
    
    community_stats.append({
        'Community': 'HONORS',
        'Interested Students': honors_interested,
        'Active Members': honors_active
    })
    
    # Handle other communities
    df_no_honors = df[df['Community Interest'] != 'Honors Community'].copy()
    df_no_honors['Community Interest'] = df_no_honors['Community Interest'].apply(standardize_community_name)
    df_no_honors['Membership'] = df_no_honors['Membership'].apply(standardize_community_name)
    
    interests = df_no_honors['Community Interest'].value_counts()
    active_memberships = df_no_honors[
        df_no_honors['Membership Status'] == 'Active'
    ]['Membership'].value_counts()
    
    all_communities = set(interests.index) | set(active_memberships.index)
    
    for community in all_communities:
        if pd.notna(community) and community != 'HONORS':
            interested = interests.get(community, 0)
            active = active_memberships.get(community, 0)
            community_stats.append({
                'Community': community,
                'Interested Students': interested,
                'Active Members': active
            })
    
    return pd.DataFrame(community_stats)
# Main Dashboard Code
def main():
    st.title("🏠 Housing Analysis Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your Peak Report CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and preprocess data
        df_original = pd.read_csv(uploaded_file)
        df, duplicates_removed = preprocess_data(df_original)
        
        if duplicates_removed > 0:
            st.warning(f"✨ Data cleaned: Removed {duplicates_removed} duplicate entries")
        
        # Create main tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Roommate Analysis",
                "👥 Community Analysis",
                "📚 GPA Analysis",
                "🛏️ DAC Analysis"
            ])

        
        # Tab 1: Roommate Analysis
        with tab1:
            st.header("Roommate Group Analysis")
            
            # Basic group statistics
            roommate_groups = df[df['Roommate Group ID'] != 0]
            group_sizes = roommate_groups['Roommate Group ID'].value_counts()
            total_groups = len(group_sizes)
            total_people = len(roommate_groups)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total People in Groups", total_people)
            with col2:
                st.metric("Number of Groups", total_groups)
            with col3:
                st.metric("Average Group Size", f"{total_people/total_groups:.2f}")
            
            # Group size distribution
            st.subheader("Group Size Distribution")
            size_distribution = group_sizes.value_counts().sort_index()
            
            dist_df = pd.DataFrame({
                'Group Size': size_distribution.index,
                'Number of Groups': size_distribution.values,
                'Total Students': size_distribution.index * size_distribution.values
            })
            
            fig_groups = px.bar(
                dist_df,
                x='Group Size',
                y='Number of Groups',
                title='Distribution of Roommate Group Sizes'
            )
            fig_groups.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_groups, use_container_width=True)
            st.dataframe(dist_df)
            
            # Compatibility Analysis
            st.header("Compatibility Analysis")

            # Calculate compatibility scores
            compatibility_scores = []
            for group_id in df[df['Roommate Group ID'] != 0]['Roommate Group ID'].unique():
                group_members = df[df['Roommate Group ID'] == group_id]
                if len(group_members) > 1:
                    for i, member1 in group_members.iterrows():
                        for j, member2 in group_members.iterrows():
                            if i < j:
                                score = analyze_compatibility(
                                    member1['Roommate Questionnaire String'],
                                    member2['Roommate Questionnaire String']
                                )
                                if score is not None:
                                    compatibility_scores.append(score)

            if compatibility_scores:
                # Calculate the average compatibility
                avg_compatibility = np.mean(compatibility_scores)
                st.metric("Average Compatibility", f"{avg_compatibility:.2f}/5")

                # Create a histogram with more attractive features
                fig_compat = px.histogram(
                    compatibility_scores,
                    nbins=6,
                    color_discrete_sequence=["#133E87"],  # Attractive color
                    title="Distribution of Roommate Compatibility Scores",
                    labels={"value": "Compatibility Score (out of 5)", "count": "Number of Pairs"}
                )

                # Update layout for better aesthetics
                fig_compat.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                    font_color='white',
                    title_font_size=20,  # Larger title
                    xaxis=dict(
                        title="Compatibility Score (out of 5)",
                        tickmode="linear",
                        tick0=0,
                        dtick=1  # Ensure each score is labeled
                    ),
                    yaxis=dict(
                        title="Number of Pairs",
                        showgrid=True
                    )
                )

                # Add annotations for percentages
                hist_values, bin_edges = np.histogram(compatibility_scores, bins=6)
                for i, count in enumerate(hist_values):
                    percentage = count / len(compatibility_scores) * 100
                    fig_compat.add_annotation(
                        x=(bin_edges[i] + bin_edges[i + 1]) / 2,  # Center of the bin
                        y=count + 10,  # Slightly above the bar
                        text=f"{percentage:.1f}%",
                        showarrow=False,
                        font=dict(size=12, color="white")
                    )

                # Show the updated plot
                st.plotly_chart(fig_compat, use_container_width=True)
        
        # Tab 2: Community Analysis
        with tab2:
            st.header("Community Interest Analysis")

            # Process the community data
            community_data = get_community_stats(df)

            # Calculate total interested and active students
            total_interested = community_data['Interested Students'].sum()
            total_active = community_data['Active Members'].sum()

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Interested Students", total_interested)
            with col2:
                st.metric("Total Active Members", total_active)

            # Prepare data for plotting
            community_df = community_data.melt(
                id_vars="Community", 
                value_vars=["Interested Students", "Active Members"],
                var_name="Status",
                value_name="Count"
            )

            # Map "Status" to more readable names
            community_df['Status'] = community_df['Status'].replace({
                "Interested Students": "Interested",
                "Active Members": "Active"
            })

            # Create the bar chart using Plotly Express
            fig = px.bar(
                community_df,
                x="Community",
                y="Count",
                color="Status",
                barmode="group",
                text="Count",  # Show values on the bars
                title="Community Interest vs Active Membership"
            )

            # Update layout for better aesthetics
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                font_color='white',  # White font for dark theme
                xaxis=dict(title="Community", tickmode="linear"),
                yaxis=dict(title="Number of Students"),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Show the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)


            
            st.subheader("Detailed Community Numbers")
            st.dataframe(community_data)
            
            st.subheader("Interest to Membership Conversion Rates")
            for _, row in community_data.iterrows():
                if row['Interested Students'] > 0:
                    conversion_rate = np.round((row['Active Members'] / row['Interested Students']) * 100, 1)
                    st.metric(
                        f"{row['Community']} Conversion Rate",
                        f"{conversion_rate}%"
                    )
                    # Tab 3: GPA Analysis
        with tab3:
            st.header("📊 GPA Analysis")
            
            # Advanced Filters Section
            st.subheader("Advanced Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender_filter = st.selectbox("Filter by Gender", 
                    get_sorted_unique_values(df, 'Gender'))
                min_gpa = st.slider("Minimum GPA", 0.0, 4.0, 0.0, 0.1)
                
            with col2:
                building_filter = st.selectbox("Filter by Building", 
                    get_sorted_unique_values(df, 'Building'))
                academic_status = st.selectbox("Academic Status",
                    get_sorted_unique_values(df, 'Academic Status'))
                    
            with col3:
                ethnicity_filter = st.selectbox("Filter by Ethnicity",
                    get_sorted_unique_values(df, 'Ethnicity'))
                class_filter = st.selectbox("Class Standing",
                    get_sorted_unique_values(df, 'Enrollment Class'))
                exclude_zero_gpa = st.checkbox("Exclude 0.0 GPA Entries", value=False)
            
            # Create initial filter dictionary
            filters = {
                'Gender': gender_filter,
                'Building': building_filter,
                'Ethnicity': ethnicity_filter,
                'Enrollment Class': class_filter,
                'Academic Status': academic_status
            }
            
            # Apply filters
            filtered_df = df.copy()
            for column, value in filters.items():
                if value != 'All':
                    filtered_df = filtered_df[filtered_df[column] == value]
            filtered_df = filtered_df[filtered_df['GPA'].notna()]
            if min_gpa > 0:
                filtered_df = filtered_df[filtered_df['GPA'] >= min_gpa]
            if exclude_zero_gpa:
                filtered_df = filtered_df[filtered_df['GPA'] > 0]
            
            # Show number of students matching criteria
            st.info(f"Showing data for {len(filtered_df)} students matching the criteria")
            
            # GPA Analysis Tabs
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                "Comparative Analysis", 
                "Community Analysis",
                "Statistical Breakdown"
            ])
            
            with analysis_tab1:
                st.subheader("GPA Comparative Analysis")
                
                # Average GPA by Gender with count
                gender_gpa = filtered_df.groupby('Gender').agg({
                    'GPA': ['mean', 'count']
                }).reset_index()
                gender_gpa.columns = ['Gender', 'Average GPA', 'Student Count']
                
                fig_gender = px.bar(
                    gender_gpa,
                    x='Gender',
                    y='Average GPA',
                    color='Student Count',
                    text=gender_gpa['Average GPA'].round(2),
                    title='Average GPA by Gender',
                    labels={'Student Count': 'Number of Students'}
                )
                fig_gender.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_gender, use_container_width=True)
                
                # GPA Distribution by Class Standing
                class_gpa = filtered_df.groupby('Enrollment Class').agg({
                    'GPA': ['mean', 'count']
                }).reset_index()
                class_gpa.columns = ['Class', 'Average GPA', 'Student Count']
                
                fig_class = px.line(
                    class_gpa,
                    x='Class',
                    y='Average GPA',
                    text=class_gpa['Average GPA'].round(2),
                    title='GPA Progression by Class Standing',
                    markers=True
                )
                fig_class.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_class, use_container_width=True)
            
            with analysis_tab2:
                st.subheader("Community GPA Analysis")
                
                # Get community data
                community_df = filtered_df[filtered_df['Community Interest'].notna()].copy()
                community_df['Standardized Community'] = community_df['Community Interest'].apply(standardize_community_name)
                
                # Calculate community GPA stats
                community_gpa = community_df.groupby('Standardized Community').agg({
                    'GPA': ['mean', 'count']
                }).reset_index()
                community_gpa.columns = ['Community', 'Average GPA', 'Student Count']
                
                # Community GPA Comparison
                fig_community = px.bar(
                    community_gpa,
                    x='Community',
                    y='Average GPA',
                    color='Student Count',
                    text=community_gpa['Average GPA'].round(2),
                    title='Average GPA by Community',
                    labels={'Student Count': 'Number of Students'}
                )
                fig_community.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_community, use_container_width=True)
                
                # First Generation Student Analysis
                firstgen_gpa = filtered_df.groupby('First Generation Student').agg({
                    'GPA': ['mean', 'count']
                }).reset_index()
                firstgen_gpa.columns = ['First Generation', 'Average GPA', 'Student Count']
                
                fig_firstgen = px.bar(
                    firstgen_gpa,
                    x='First Generation',
                    y='Average GPA',
                    text=firstgen_gpa['Average GPA'].round(2),
                    title='GPA Comparison: First Generation vs Non-First Generation',
                    color='Student Count'
                )
                fig_firstgen.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_firstgen, use_container_width=True)
            
            with analysis_tab3:
                st.subheader("Statistical Summary")
                
                # Overall GPA Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average GPA", f"{filtered_df['GPA'].mean():.2f}")
                with col2:
                    st.metric("Median GPA", f"{filtered_df['GPA'].median():.2f}")
                with col3:
                    st.metric("Total Students", len(filtered_df))
                
                # GPA Range Distribution
                gpa_ranges = pd.cut(filtered_df['GPA'], 
                                  bins=[0, 2.0, 2.5, 3.0, 3.5, 4.0],
                                  labels=['0-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0'])
                
                range_counts = gpa_ranges.value_counts().sort_index()
                
                # Create detailed statistics table
                stats_df = pd.DataFrame({
                    'GPA Range': range_counts.index,
                    'Number of Students': range_counts.values,
                    'Percentage': (range_counts.values / len(filtered_df) * 100).round(1)
                })
                
                st.dataframe(stats_df)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Filtered Data",
                        csv_filtered,
                        "filtered_gpa_data.csv",
                        "text/csv",
                        key='download-filtered'
                    )
                
                with col2:
                    stats_df = filtered_df.groupby(['Gender', 'Building', 'Enrollment Class'])['GPA'].agg([
                        'count', 'mean', 'median'
                    ]).round(2).reset_index()
                    csv_stats = stats_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download Summary Statistics",
                        csv_stats,
                        "gpa_summary_stats.csv",
                        "text/csv",
                        key='download-stats'
                    )
                    # DAC Analysis
            with tab4:
                st.header("🛏️ DAC Analysis")

                # Check for required columns
                required_columns = ["DAC Recommendation Received", "Returner Status", "Single", "Received from DAC", "Room Type", "Meal Plan Waiver"]
                if all(col in df.columns for col in required_columns):
                    # Filter for people with DAC Recommendations
                    dac_received = df[df["DAC Recommendation Received"] == True]

                    # Convert 'Received from DAC' to datetime, handling errors
                    dac_received["Received from DAC"] = pd.to_datetime(dac_received["Received from DAC"], errors='coerce')

                    # 1. DAC Recommendation Analysis
                    st.subheader("DAC Recommendation Analysis")

                    dac_true_count = dac_received.shape[0]
                    #dac_false_count = df[df["DAC Recommendation Received"] == False].shape[0]

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("DAC Recommendation Received", dac_true_count)
                    #with col2:
                        #st.metric("People without Accommodation Recommendations", dac_false_count)

                    # Pie chart for DAC recommendation status
                    #fig_dac = px.pie(
                       # values=[dac_true_count, dac_false_count],
                       ## title="DAC Recommendation Status",
                       # color_discrete_sequence=px.colors.sequential.RdBu
                  #  )
                    #fig_dac.update_layout(
                    #    plot_bgcolor='rgba(0,0,0,0)',
                   #     paper_bgcolor='rgba(0,0,0,0)',
                    #    font_color='white'
                   # )
                   # st.plotly_chart(fig_dac, use_container_width=True)

                   # 2. Returner vs New Students Analysis
                    st.subheader("Returner vs New Students with DAC Recommendation")

                    # Assuming "Classification Description" has values like "Returning" and "New"
                    returners_with_dac = dac_received[dac_received["Classification Description"] == "Returner"]
                    new_students_with_dac = dac_received[dac_received["Classification Description"] != "Returner"]

                    # Display metrics
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Returning Students with DAC", len(returners_with_dac))
                    with col4:
                        st.metric("New Students with DAC", len(new_students_with_dac))

                    # Pie chart for returners vs new students
                    fig_returners = px.pie(
                        values=[len(returners_with_dac), len(new_students_with_dac)],
                        names=["Returning Students", "New Students"],
                        title="Comparison: Returning vs New Students with DAC Recommendation",
                        color_discrete_sequence=["#0A5EB0", "#C9E6F0"]
                    )
                    fig_returners.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_returners, use_container_width=True)

                    # Interactive Student Selection with Buttons
                    #st.subheader("Student Details with DAC Recommendation")

                    # Columns for buttons
                    col1, col2 = st.columns(2)

                    # Button for Returning Students
                    with col1:
                        if st.button("View Returning Students Data"):
                            st.dataframe(returners_with_dac[["Name", "Student Number", "DAC Recommendation Received", "Classification Description"]])

                    # Button for New Students
                    with col2:
                        if st.button("View New Students Data"):
                            st.dataframe(new_students_with_dac[["Name", "Student Number", "DAC Recommendation Received", "Classification Description"]])

                    # 3. Single Accommodation Analysis
                    st.subheader("Single Accommodation Needs")

                    # Filter for students needing single accommodation
                    single_needed = dac_received[dac_received["Single"] == True]

                    # Identify room types containing "Single" (case-insensitive)
                    single_needed["Room Type (Single)"] = single_needed["Room Type"].str.contains("Single", case=False, na=False)

                    # Separate assigned and not assigned
                    single_assigned = single_needed[single_needed["Room Type (Single)"]]
                    single_not_assigned = single_needed[~single_needed["Room Type (Single)"]]

                    # Calculate counts
                    total_single_needed = single_needed.shape[0]
                    assigned_single_count = single_assigned.shape[0]
                    not_assigned_single_count = single_not_assigned.shape[0]

                    # Display metrics
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("Total Students Needing Single Accommodation", total_single_needed)
                    with col6:
                        st.metric("Students Assigned Single Rooms", assigned_single_count)
                    with col7:
                        st.metric("Students Not Assigned Single Rooms", not_assigned_single_count)

                    # Display table for students not assigned single rooms
                    if not_assigned_single_count > 0:
                        st.subheader("Details of Students Not Assigned Single Rooms")
                        st.dataframe(single_not_assigned[["Name", "Room Type", "Single"]])
                    else:
                        st.info("All students needing single accommodations are assigned to single rooms.")

                    # 4. Approved Accommodations Before June 31
                    st.subheader("Approved Accommodations Before June 31")

                    # Filter dates before June 31
                    june_30_date = pd.Timestamp(year=2024, month=6, day=30)
                    approved_before_june_31 = dac_received[dac_received["Received from DAC"] <= june_30_date].shape[0]
                    approved_after_june_31 = dac_received[dac_received["Received from DAC"] > june_30_date].shape[0]

                    # Display metric
                    st.metric("Approved Accommodations Before June 31", approved_before_june_31)

                    # Bar chart for timing of accommodations
                    timing_counts = pd.DataFrame({
                        "Approval Timing": ["Before June 31", "After June 31"],
                        "Count": [approved_before_june_31, approved_after_june_31]
                    })

                    fig_timing_bar = px.bar(
                        timing_counts,
                        x="Approval Timing",
                        y="Count",
                        text="Count",
                        title="Accommodation Approvals: Before vs After June 31",
                        labels={"Approval Timing": "Timing", "Count": "Number of Approvals"},
                        color="Approval Timing",
                        color_discrete_sequence=["#ADD8E6", "#1E90FF"]
                    )
                    fig_timing_bar.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_timing_bar, use_container_width=True)
                    # 5. Meal Plan Waiver Analysis
                    st.subheader("Meal Plan Waiver Analysis")

                    # Count the number of students with meal plan waivers
                    meal_plan_waiver_count = df[df["Meal Plan Waiver"] == True].shape[0]

                    # Display metrics for students with meal plan waivers only
                    st.metric("Students with Meal Plan Waiver", meal_plan_waiver_count)

                else:
                    st.error(f"Required columns {required_columns} are not available in the dataset.")
    else:
        st.info("Please upload your CSV file to begin the analysis")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input data and try again.")
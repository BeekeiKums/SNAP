import csv, openpyxl
import pandas as pd
from apify_client import ApifyClient
from django.http import HttpResponse
from io import TextIOWrapper
from django.contrib.auth.decorators import login_required
from neomodel import db
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from .models import Category
from django.contrib import messages
from .forms import UserAccountForm, CategoryForm, BusinessmanForm , ContentCreatorForm, DataAnalystForm , DataForm 
from .models import UserAccount , ExtractData , Data , SocialMediaData
from django.http import JsonResponse
from django.conf import  settings
from .models import Person, Movie
from django.views.decorators.csrf import csrf_exempt







def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user_account = UserAccount.objects.get(username=username)
            if user_account.password == password:  # This is fine if not using Django's auth
                # Set session for login
                request.session['username'] = user_account.username
                request.session['role'] = user_account.role

                # Redirect to respective dashboards based on role
                if user_account.role == 'admin':
                    return redirect('dashboard')
                elif user_account.role == 'businessman':
                    return redirect('businessman_dashboard')
                elif user_account.role == 'content_creator':
                    return redirect('content_creator_dashboard')
                elif user_account.role == 'data_analyst':
                    return redirect('data_analyst_dashboard')
            else:
                messages.error(request, "Invalid credentials!")
        except UserAccount.DoesNotExist:
            messages.error(request, "Account does not exist!")

    return render(request, 'main/login.html')


        

def admin_logout(request):
    logout(request)
    return redirect('admin_login')


def businessman_dashboard(request):
    return render(request, 'main/businessman.html')


def content_creator_dashboard(request):
    return render(request, 'main/content_creator.html')


def data_analyst_dashboard(request):
    return render(request, 'main/data_analyst.html')


# Dashboard View
def dashboard(request):
    return render(request, 'main/dashboard.html')



# Create Category
def create_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save() 
            return redirect('category_list')
    
    else: 
        form = CategoryForm()  #Empty form for GET requet
    return render(request, 'main/create_category.html', {'form': form}) 


@csrf_exempt
def update_category(request, category_id):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')

        try:
            category = Category.objects.get(id=category_id)
            category.name = name
            category.description = description
            category.save()

            # Return both the updated name and description
            return JsonResponse({
                'status': 'success',
                'id': category.id,
                'name': category.name,
                'description': category.description
            })
        except Category.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Category not found'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
  
    
def category_list(request):
    categories = Category.objects.all()  # Make sure this is fetching the categories correctly
    return render(request, 'main/category_list.html', {'categories': categories})


#create user account      
def create_user_account(request):
    if request.method == 'POST':
        form = UserAccountForm(request.POST)
        if form.is_valid():
            # Check for duplicate usernames or emails
            
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else:
                # Save the user account
                form.save()
                messages.success(request, "User account created successfully!")
                return redirect('create_user_account')  # Redirect to success page

        else:
            messages.error(request, "Failed to create account. Please check your input.")
    else:
        form = UserAccountForm()

    return render(request, 'main/create_user_account.html', {'form': form})

def create_businessman_account(request):
    if request.method == 'POST':
        form = BusinessmanForm(request.POST)
        if form.is_valid():
            # Check for duplicate usernames or emails
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else:
                # Save the user account and set role to 'businessman'
                user_account = form.save(commit=False)  # Do not save to DB yet
                user_account.role = 'businessman'       # Assign role in backend
                user_account.save()                    # Save to DB
                messages.success(request, "Businessman account created successfully!")
                return redirect('create_businessman_account')  # Redirect back to the form
    else:
        form = BusinessmanForm()
    return render(request, 'main/create_businessman_acc.html', {'form': form})


def create_content_creator_account(request):
    if request.method == 'POST':
        form = ContentCreatorForm(request.POST)
        if form.is_valid():
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else: 
                user_account = form.save(commit=False)
                user_account.role = 'content_creator'
                user_account.save()
                
                messages.success(request, "Content cretor account created successfully!")
                return redirect('create_content_creator_account')
    else :
        form = ContentCreatorForm()
    return render(request, 'main/create_content_creator_acc.html', {'form': form}) 

def create_data_analyst_account(request):
    if request.method == 'POST':
        form = DataAnalystForm(request.POST)
        if form.is_valid():
            if UserAccount.objects.filter(username=form.cleaned_data['username']).exists():
                messages.error(request, "Username already exists!")
            elif UserAccount.objects.filter(email=form.cleaned_data['email']).exists():
                messages.error(request, "Email already exists!")
            else :
                user_account = form.save(commit=False)
                user_account.role = 'data_analyst'
                user_account.save()
                messages.success(request, "Data analyst account created successfully!")
                return redirect('create_data_analyst_account')
    else :
        form = DataAnalystForm()
    return render(request, 'main/create_data_analyst_acc.html', {'form': form})                    
                           

#View to list all user accounts
def view_user_accounts(request):
    users = UserAccount.objects.all()
    return render(request, 'main/view_user_accounts.html', {'users': users})

def view_businessman_accounts(request):
    users = UserAccount.objects.filter(role  = 'businessman')
    return render(request, 'main/view_businessman_accounts.html', {'users': users})

def view_content_creator_accounts(request):
    users = UserAccount.objects.filter(role = 'content_creator')
    return render(request, 'main/view_content_accounts.html', {'users': users})

def view_data_analyst_accounts(request):
    users = UserAccount.objects.filter(role = 'data_analyst')
    return render(request, 'main/view_data_analyst_accounts.html', {'users':users})



def update_user_account(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(UserAccount, id=user_id)
        user.username = request.POST.get('username')
        user.email = request.POST.get('email')
        user.save()
        
        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'fail'})


import os
from dotenv import load_dotenv


# Extract Data
            





from django.http import HttpResponse
import csv



#upload csv

import csv
from io import TextIOWrapper
from django.shortcuts import render, redirect
from django.contrib import messages


'''
def upload_csv_or_xlsx(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        
        if not uploaded_file:
            messages.error(request, "Error: No file uploaded.")
            return redirect('upload_csv_or_xlsx')
        
        try:
            #Save the original file name in the session
            original_file_name = os.path.splitext(uploaded_file.name)[0] #Remove extension
            cleaned_file_name = original_file_name.strip() + ".csv" #Ensure '.csv' extension
            request.session['original_file_name'] = cleaned_file_name
            
            
             
            
            
            # Read CSV file
            file_data = TextIOWrapper(uploaded_file.file, encoding='utf-8')
            csv_reader = csv.DictReader(file_data)
            headers = csv_reader.fieldnames
            data = [row for row in csv_reader]

            if not headers:
                messages.error(request, "The CSV file must have at least one column.")
                return redirect('upload_csv_or_xlsx')

            # Save headers and data for later use
            request.session['uploaded_headers'] = headers
            request.session['uploaded_data'] = data
            messages.success(request, f"File {cleaned_file_name} uploaded successfully.")
            return redirect('auto_preprocess')
        except Exception as e:
            messages.error(request, f"Error processing the file: {str(e)}")
            return redirect('upload_csv_or_xlsx')
    
    return render(request, "main/upload_excel.html")
'''

import os
import csv
from io import TextIOWrapper
from django.shortcuts import render, redirect
from django.contrib import messages

def upload_csv_or_xlsx(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            messages.error(request, "Error: No file uploaded.")
            return redirect('upload_csv_or_xlsx')

        try:
            # Save the original file name in the session
            original_file_name = os.path.splitext(uploaded_file.name)[0]
            cleaned_file_name = original_file_name.strip() + ".csv"
            request.session['original_file_name'] = cleaned_file_name

            # Read CSV file
            file_data = TextIOWrapper(uploaded_file.file, encoding='utf-8')
            df = pd.read_csv(file_data)

            # Clean and convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric where possible
            
            # Replace NaNs with a placeholder or drop them
            df = df.dropna(axis=0, how='all')  # Drop rows where all values are NaN

            # Extract headers and data for later use
            headers = df.columns.tolist()
            data = df.to_dict(orient='records')  # Convert cleaned DataFrame to a list of dictionaries

            if not headers:
                messages.error(request, "The CSV file must have at least one column.")
                return redirect('upload_csv_or_xlsx')

            # Save headers and data to session
            request.session['uploaded_headers'] = headers
            request.session['uploaded_data'] = data
            messages.success(request, f"File {cleaned_file_name} uploaded successfully.")
            return redirect('auto_preprocess')

        except Exception as e:
            messages.error(request, f"Error processing the file: {str(e)}")
            return redirect('upload_csv_or_xlsx')

    return render(request, "main/upload_excel.html")


#auto_data process

import pandas as pd

#use
'''
def auto_preprocess(request):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            if not uploaded_data:
                messages.error(request, "No data available for preprocessing. Please upload a file first.")
                return redirect('upload_csv_or_xlsx')

            # Convert session data to DataFrame
            df = pd.DataFrame(uploaded_data)

            # Step 1: Drop rows and columns where all values are NaN
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # Step 2: Clean column names dynamically
            df.columns = (
                df.columns.str.strip()
                .str.replace(r'\W+', '_', regex=True)
                .str.lower()
            )
            
            # Remove columns with a single unique value (constant columns)
            df = df.loc[:, df.nunique() > 1]

            # Remove columns with > 90% missing values
            df = df.loc[:, df.isnull().mean() < 0.9]

            # Step 3: Process columns dynamically
            processed_columns = []
            ignored_columns = []
            label_encoder = LabelEncoder()  # For encoding categories
            max_unique_values = 50  # Limit unique values to avoid graph clutter

            for col in df.columns:
                if df[col].dtype == 'object':  # Handle text columns
                    # Remove unnecessary whitespace and invalid strings
                    df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'NaN': '', 'None': ''})

                    # If possible, convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # If still non-numeric, encode the column
                    if df[col].isnull().all():  # Still non-numeric
                        df[col] = label_encoder.fit_transform(df[col].astype(str))
                        processed_columns.append(col)

                elif pd.api.types.is_numeric_dtype(df[col]):  # Handle numeric columns
                    df[col].fillna(0, inplace=True)
                    processed_columns.append(col)

                else:
                    ignored_columns.append(col)

            # Step 4: Sample rows for performance
            if len(df) > 500:  # Adjust sampling threshold as needed
                df = df.sample(n=500, random_state=42)

            # Save cleaned data back to the session
            request.session['uploaded_data'] = df.to_dict(orient='records')
            request.session['uploaded_headers'] = df.columns.tolist()

            messages.success(request, "Data successfully preprocessed. All columns converted to numeric where possible.")
            return redirect('manual_preprocess')

        except Exception as e:
            messages.error(request, f"Error during preprocessing: {str(e)}")
            return redirect('upload_csv_or_xlsx')

    return render(request, 'main/data_management.html')

'''

import re
def process_columns(dataframe):
    new_columns = {}

    for col in dataframe.columns:
        try:

            dataframe[col] = dataframe[col].astype(float)

            if 'count' not in col.lower():
                new_columns[col] = f"{col}_count"
        except ValueError:
            
            pass

    
    dataframe = dataframe.rename(columns=new_columns)
    
    return dataframe

def func(dataframe):
    analyze_lists = [r'.*view.*',r'^like.*count',r'comment.*count',r'share.*count']
    column_names = ['view_count','like_count','comment_count','share_count']
    df_test = pd.DataFrame()
    if not any(dataframe.columns.str.contains(r'(?i)hashtag')):
    
        df_test['hashtags'] = pd.Series(['-']*len(dataframe))
        for index, regex in enumerate(analyze_lists):
            df_test[column_names[index]] = pd.Series([0.0]*len(dataframe))

            view_columns = [col for col in dataframe.columns if re.match(regex, col, re.IGNORECASE)]
            
            for i in view_columns:
                dataframe[i] = dataframe[i].fillna(0)
                if dataframe[i].dtype == 'int' or dataframe[i].dtype == 'float':
                    df_test[column_names[index]] += dataframe[i]  
    else:
        hashtag_columns = [col for col in dataframe.columns if 'hashtag' in col.lower() and dataframe[col].dtypes == object]
        
    
        columns_to_split = [col for col in hashtag_columns if dataframe[col].str.contains(',').all()]
        for col in columns_to_split:
    
            split_cols = dataframe[col].str.split(',', expand=True)
    
            for i in range(split_cols.shape[1]):
                dataframe[f'{col}_text{i+1}'] = split_cols[i]
    
            dataframe = dataframe.drop(columns=col)

        hashtag_columns = [col for col in dataframe.columns if 'hashtag' in col.lower()]
        hashtag_columns_length = dataframe[hashtag_columns].notnull().sum(axis=1).tolist()
        hashtag_values = dataframe[hashtag_columns].values.tolist()

        all_hashtags = [
            tag  if len(tag) < 100 else 'link'
            for tags in hashtag_values
            for tag in tags
            if isinstance(tag, str)
        ]
        
        
        df_test['hashtags'] = pd.Series(all_hashtags)

        hashtag_columns = [col for col in dataframe.columns if 'hashtag' in col.lower() and dataframe[col].dtypes == object]
        hashtag_columns_length = dataframe[hashtag_columns].notnull().sum(axis=1).tolist()
        
        for index, regex in enumerate(analyze_lists):
            df_test[column_names[index]] = pd.Series([0.0]*len(all_hashtags))
            view_columns = [col for col in dataframe.columns if re.match(regex, col, re.IGNORECASE)]
        
            for i in view_columns:
                dataframe[i] = dataframe[i].fillna(0)
                if dataframe[i].dtype == 'int' or dataframe[i].dtype == 'float':
                    df_test[column_names[index]] += dataframe[i]

        
            all_views = [count / length for count, length in zip(df_test[column_names[index]], hashtag_columns_length) for _ in range(length)]
            df_test[column_names[index]] = pd.Series(all_views)

    return df_test


def auto_preprocess(request):
    import pandas as pd

    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            if not uploaded_data:
                messages.error(request, "No data available for preprocessing. Please upload a file first.")
                return redirect('upload_csv_or_xlsx')

            # Convert session data to DataFrame
            df = pd.DataFrame(uploaded_data)

           
            
            
            df = process_columns(df)
            df = func(df)
            processed_columns = df.columns.tolist()
            

            # Save cleaned data back to the session
            request.session['uploaded_data'] = df.to_dict(orient='records')
            request.session['uploaded_headers'] = df.columns.tolist()

            messages.success(request, "Data successfully preprocessed. All columns converted to numeric where possible.")
            return redirect('manual_preprocess')

        except Exception as e:
            messages.error(request, f"Error during preprocessing: {str(e)}")
            return redirect('upload_csv_or_xlsx')

    return render(request, 'main/data_management.html')



'''           
def manual_preprocess(request):
    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            headers = request.session.get('uploaded_headers', [])

            # Process form submission
            for index, row in enumerate(uploaded_data):
                for header in headers:
                    # Update each value in the row based on form input
                    field_name = f'data_{index}_{header}'
                    row[header] = request.POST.get(field_name, row.get(header, ''))

            # Update session data with modified values
            request.session['uploaded_data'] = uploaded_data
            messages.success(request, "Data successfully updated.")

            return redirect('create_or_view_visualization')  # Redirect to next step
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('manual_preprocess')  # Redirect back to manual preprocess page on error

    # Retrieve preprocessed data for display
    uploaded_data = request.session.get('uploaded_data', [])
    headers = request.session.get('uploaded_headers', [])

    return render(request, 'main/manual.html', {'data': uploaded_data, 'headers': headers})
'''

def manual_preprocess(request):
    import math

    # Pagination settings
    PAGE_SIZE = 100  # Number of rows per page
    if request.method == 'POST':
        try:
            # Retrieve uploaded data from the session
            uploaded_data = request.session.get('uploaded_data', [])
            headers = request.session.get('uploaded_headers', [])

            # Update rows with submitted data
            for index, row in enumerate(uploaded_data):
                for header in headers:
                    field_name = f'data_{index}_{header}'
                    row[header] = request.POST.get(field_name, row.get(header, ''))

            # Save updated data back to the session
            request.session['uploaded_data'] = uploaded_data
            messages.success(request, "Data successfully updated.")
            return redirect('create_or_view_visualization')
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('manual_preprocess')

    # Retrieve data for display
    uploaded_data = request.session.get('uploaded_data', [])
    headers = request.session.get('uploaded_headers', [])
    
    # Implement pagination
    page = int(request.GET.get('page', 1))
    total_pages = math.ceil(len(uploaded_data) / PAGE_SIZE)
    paginated_data = uploaded_data[(page -1) * PAGE_SIZE : page * PAGE_SIZE]

    return render(request, 'main/manual.html', {'data': paginated_data, 
                                                'headers': headers,
                                                'current_page': page,
                                                'total_pages': total_pages})

  
        
# views.py

from django.shortcuts import render, redirect
from django.contrib import messages
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

'''
def dashboard_view(request):
    
    #Retrieve headers from session
    headers = request.session.get('uploaded_headers', [])
    
    # Display dashboard view with options for data visualization
    return render(request, 'main/dash_visualization.html',{
        'headers': headers
    })
'''


def dashboard_view(request):
    # Retrieve headers and data from session
    headers = request.session.get('uploaded_headers', [])
    uploaded_data = request.session.get('uploaded_data', [])

    if not headers or not uploaded_data:
        messages.error(request, "No data uploaded. Please upload a file first.")
        return redirect('upload_csv_or_xlsx')

    # Filter numeric columns only
    df = pd.DataFrame(uploaded_data)
    numeric_headers = [col for col in headers if pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce'))]

    return render(request, 'main/dash_visualization.html', {
        'headers': numeric_headers
    })







'''
def create_or_view_visualization(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for visualization.")
            return redirect('upload_csv_or_xlsx')
        
        
        
        # Step 1: Build the Graph
        G = nx.Graph()
        

        for row in uploaded_data:
            row_nodes = []
            for header in headers:
                value = row.get(header)
                # Filter out noisy or invalid values
                if value and isinstance(value, str) and len(value) < 50 and value not in ['null', 'None', 'nan', '-', '']:
                    # Exclude URLs and timestamps
                    if 'http' in value or 'timestamp' in header.lower():
                        continue
                    G.add_node(value)
                    row_nodes.append(value)
            
            # Dynamically limit edges between nodes to avoid clutter
            max_edges_per_node = 10
            node_edges_count = {}
            for i in range(len(row_nodes)):
                for j in range(i + 1, len(row_nodes)):
                    node1, node2 = row_nodes[i], row_nodes[j]
                    # Skip if the nodes already have too many edges
                    if node_edges_count.get(node1, 0) >= max_edges_per_node:
                        continue
                    if node_edges_count.get(node2, 0) >= max_edges_per_node:
                        continue
                    G.add_edge(node1, node2)
                    node_edges_count[node1] = node_edges_count.get(node1, 0) + 1
                    node_edges_count[node2] = node_edges_count.get(node2, 0) + 1
                    
        save_to_neo4j(G, headers)            

        # Step 2: Filter the Graph for Large Datasets
        if G.number_of_nodes() > 200:  # Threshold for large datasets
            degree_cutoff = 2  # Remove nodes with degree <= 2
            nodes_to_keep = [node for node, degree in G.degree() if degree > degree_cutoff]
            G = G.subgraph(nodes_to_keep).copy()
            
            

        # Step 3: Dynamic Layout
        layout_style = request.GET.get('layout', 'kamada_kawai')
        if layout_style == 'spring':
            pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust "k" for spacing
        elif layout_style == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai for better large graph visualization

        # Step 4: Highlight High-Degree Nodes
        degree_threshold = 5  # Nodes with degree > 5 are important
        high_degree_nodes = [node for node, degree in G.degree() if degree > degree_threshold]
        node_colors = ['orange' if node in high_degree_nodes else 'skyblue' for node in G.nodes]
        node_sizes = [max(200, min(degree * 50, 1000)) for _, degree in G.degree()]

        # Step 5: Plot the Graph
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, with_labels=True, 
                node_color=node_colors, 
                node_size=node_sizes, 
                edge_color="gray", 
                font_size=8, alpha=0.7)
        
        # Save image to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Render visualization
        return render(request, 'main/view_visualization.html', {
            'headers' : headers,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'visualization_image': f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        messages.error(request, f"Error creating visualization: {str(e)}")
        return redirect('upload_csv_or_xlsx')
'''

#use
'''
def create_or_view_visualization(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for visualization.")
            return redirect('upload_csv_or_xlsx')

        # Step 1: Build the Graph
        G = nx.Graph()
        for row in uploaded_data:
            row_nodes = []
            for header in headers:
                value = row.get(header)
                if value and isinstance(value, str) and len(value) < 50 and value not in ['null', 'None', 'nan', '-', '']:
                    if 'http' in value or 'timestamp' in header.lower():
                        continue
                    G.add_node(value)
                    row_nodes.append(value)

            # Add edges with dynamic limit
            max_edges_per_node = 10
            node_edges_count = {}
            for i in range(len(row_nodes)):
                for j in range(i + 1, len(row_nodes)):
                    node1, node2 = row_nodes[i], row_nodes[j]
                    if node_edges_count.get(node1, 0) >= max_edges_per_node:
                        continue
                    if node_edges_count.get(node2, 0) >= max_edges_per_node:
                        continue
                    G.add_edge(node1, node2)
                    node_edges_count[node1] = node_edges_count.get(node1, 0) + 1
                    node_edges_count[node2] = node_edges_count.get(node2, 0) + 1

        # Step 2: Calculate Centrality Metrics
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        # Combine centrality measures into a dictionary
        centrality_metrics = {}
        for node in G.nodes():
            centrality_metrics[node] = {
                "degree": degree_centrality.get(node, 0),
                "closeness": closeness_centrality.get(node, 0),
                "betweenness": betweenness_centrality.get(node, 0)
            }

        # Identify top nodes by degree centrality
        top_degree_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]

        # Step 3: Dynamic Layout
        layout_style = request.GET.get('layout', 'kamada_kawai')
        if layout_style == 'spring':
            pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust "k" for spacing
        elif layout_style == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai for better large graph visualization


            


        # Step 4: Highlight Nodes Based on Centrality
        node_sizes = [max(200, degree_centrality[node] * 2000) for node in G.nodes()]
        node_colors = ['orange' if node in top_degree_nodes else 'skyblue' for node in G.nodes()]

        # Step 5: Plot the Graph
        plt.figure(figsize=(16, 12))
        nx.draw(G, pos, with_labels=True, 
                node_color=node_colors, 
                node_size=node_sizes, 
                edge_color="gray", 
                font_size=8, alpha=0.7)

        # Save image to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Prepare centrality results for frontend display
        centrality_results = [
            {"node": node, "degree": round(metrics["degree"], 4), 
             "closeness": round(metrics["closeness"], 4), 
             "betweenness": round(metrics["betweenness"], 4)}
            for node, metrics in sorted(centrality_metrics.items(), key=lambda x: x[1]["degree"], reverse=True)[:10]
        ]

        # Render visualization and centrality metrics
        return render(request, 'main/view_visualization.html', {
            'headers' : headers,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'visualization_image': f"data:image/png;base64,{img_base64}",
            'centrality_results': centrality_results  # Top nodes with centrality scores
        })

    except Exception as e:
        messages.error(request, f"Error creating visualization: {str(e)}")
        return redirect('upload_csv_or_xlsx')
'''


import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from django.shortcuts import render, redirect
from django.contrib import messages
import io
import base64

def create_or_view_visualization(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for visualization.")
            return redirect('upload_csv_or_xlsx')
        
        df = pd.DataFrame(uploaded_data)

        line_fig = px.line(
        df,
        x='hashtags',
        y=['view_count', 'like_count', 'comment_count', 'share_count'],
        title="Line Graph of Social Media Metrics",
        labels={"hashtags": "Hashtags", "value": "Count"}
        )

        line_html = line_fig.to_html(full_html=False)

        
        # Bar Chart
        bar_fig = px.bar(
        df,
        x='hashtags',
        y=['view_count', 'like_count', 'comment_count', 'share_count'],
        title="Bar Chart of Social Media Metrics"
        )
        bar_html = bar_fig.to_html(full_html=False)

        return render(request, 'main/view_visualization.html', {'line_graph': line_html, 'bar_graph': bar_html})

    except Exception as e:
        messages.error(request, f"Error creating visualization: {str(e)}")
        return redirect('upload_csv_or_xlsx')




import plotly.graph_objects as go

'''
def test_predictive_models(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])
        print(request.session.get('uploaded_data'))


        if not uploaded_data or not headers:
            messages.error(request, "No data available for model testing. Please upload data.")
            return redirect('upload_csv_or_xlsx')

        # Load data into DataFrame
        df = pd.DataFrame(uploaded_data)
        print(df.dtypes)
        print(df.columns)


        # User inputs from form
        target_column = request.GET.get('target_column')
        model_type = request.GET.get('model_type', 'linear_regression')

        # Input validation
        if not target_column:
            messages.error(request, "Please select a target column.")
            return redirect('create_or_view_visualization')

        # Check if target column exists in the original dataframe
        if target_column not in df.columns:
            messages.error(request, "Selected target column does not exist.")
            return redirect('create_or_view_visualization')

        # Select only numeric columns (excluding the non-numeric target column)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
            messages.error(request, f"The selected target column '{target_column}' is not numeric. Please select a numeric column.")
            return redirect('create_or_view_visualization')

        # Define features and target
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]

        # Check if there are features left after dropping the target
        if X.empty:
            messages.error(request, "No numeric features left for modeling after excluding the target column.")
            return redirect('create_or_view_visualization')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run selected model
        metrics = {}
        if model_type == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {"Mean Squared Error": mean_squared_error(y_test, predictions), "R2 Score": r2_score(y_test, predictions)}

        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {"Mean Squared Error": mean_squared_error(y_test, predictions), "R2 Score": r2_score(y_test, predictions)}

        else:
            messages.error(request, "Invalid model type selected.")
            return redirect('create_or_view_visualization')
        print(f"Received target column: {target_column}")
        print(f"Model type: {model_type}")
        print(f"Columns in DataFrame: {df.columns}")


        # Plot the predictions
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values[:50], label="Actual", marker='o')
        plt.plot(predictions[:50], label="Predicted", marker='x')
        plt.legend()
        plt.title(f"{model_type.replace('_', ' ').title()} - Predicted vs Actual")
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Render the predictive model results
        return render(request, 'main/predictive.html', {
            'metrics': metrics,
            'prediction_plot': f"data:image/png;base64,{img_base64}",
            'model_type': model_type
        })

    except Exception as e:
        messages.error(request, f"Model testing failed: {str(e)}")
        return redirect('create_or_view_visualization')
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def test_predictive_models(request):
    try:
        # Retrieve uploaded data
        uploaded_data = request.session.get('uploaded_data', [])
        headers = request.session.get('uploaded_headers', [])

        if not uploaded_data or not headers:
            messages.error(request, "No data available for model testing. Please upload data.")
            return redirect('upload_csv_or_xlsx')

        # Load data into DataFrame
        df = pd.DataFrame(uploaded_data)
        print("Original DataFrame:")
        print(df.dtypes)
        print(df.head())
        
        

        # User inputs from form
        target_column = request.GET.get('target_column')
        model_type = request.GET.get('model_type', 'linear_regression')

        # Input validation: Target column exists
        if not target_column or target_column not in df.columns:
            messages.error(request, "Please select a valid target column.")
            return redirect('create_or_view_visualization')
        
        # Step 1: Convert target column to numeric
        df = process_columns(df)
        df = func(df)

        # Step 2: Handle non-numeric features (encode them)
        numeric_df = df.select_dtypes(['int', 'float']).copy()

        # Drop rows with missing target values
        

        # Define features and target
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        
        

        # Check if there are valid features left
        if X.empty or y.empty:
            messages.error(request, "No valid features left for modeling after preprocessing.")
            return redirect('create_or_view_visualization')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run selected model
        metrics = {}
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor()
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor()   
        else:
            messages.error(request, "Invalid model type selected.")
            return redirect('create_or_view_visualization')

        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "Mean Squared Error": mean_squared_error(y_test, predictions),
            "R2 Score": r2_score(y_test, predictions)
        }

        # Plot the predictions
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values[:50], label="Actual", marker='o')
        plt.plot(predictions[:50], label="Predicted", marker='x')
        plt.legend()
        plt.title(f"{model_type.replace('_', ' ').title()} - Predicted vs Actual")
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Render the predictive model results
        return render(request, 'main/predictive.html', {
            'metrics': metrics,
            'prediction_plot': f"data:image/png;base64,{img_base64}",
            'model_type': model_type
        })

    except Exception as e:
        messages.error(request, f"Model testing failed: {str(e)}")
        return redirect('create_or_view_visualization')

from neo4j import GraphDatabase

    
def save_to_neo4j(graph, headers):

    import logging
    #Neo4j connection details
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "bin754826" 
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def add_to_neo4j(tx):
        query = """
MATCH (h:Hashtag)-[:POSTED]->(p:Post)
RETURN 
  h.name AS hashtags, 
  SUM(p.view_count) AS view_count, 
  SUM(p.like_count) AS like_count,
  SUM(p.comment_count) AS comment_count,
  SUM(p.share_count) AS share_count
        """
        
        tx.run(query)
        
    with driver.session() as session:
        for edge in graph.edges:
            session.write_transaction(add_to_neo4j, edge[0], edge[1])
            
    driver.close()   
    logger.info("All data saved to Neo4j successfully.")
                                        
#----------------------------GRAPH VISUALIZATION----------------------------


#download csv
def download_csv(request):
    # Retrieve uploaded data and original file name from the session
    uploaded_data = request.session.get('uploaded_data', None)
    headers = request.session.get('uploaded_headers', None)
    original_file_name = request.session.get('original_file_name', "uploaded_data.csv")  # Default if no name is found
    
    if not uploaded_data or not headers:
        messages.error(request, "No uploaded data available for download. Please upload a file first.")
        return redirect('upload_csv_or_xlsx')
    
    # Ensure the file has a .csv extension
    if not original_file_name.endswith('.csv'):
        original_file_name += '.csv'

    # Create the HTTP response with the correct content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{original_file_name}"'

    
    # Write uploaded data to CSV
    writer = csv.DictWriter(response, fieldnames=headers)
    writer.writeheader()
    writer.writerows(uploaded_data)

    return response









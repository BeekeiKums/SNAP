from django.urls import path
from . import views


urlpatterns = [

    path('admin_login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    #dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    path('businessman_dashboard/', views.businessman_dashboard, name='businessman_dashboard'),
    path('content-creator-dashboard/', views.content_creator_dashboard, name='content_creator_dashboard'),
    path('data-analyst-dashboard/', views.data_analyst_dashboard, name='data_analyst_dashboard'),

    #create acc
    path('create_user_account/', views.create_user_account, name='create_user_account'),
    path('create_businessman_account/', views.create_businessman_account,  name='create_businessman_account'),
    path('create_content_creator_account/', views.create_content_creator_account, name='create_content_creator_account'),
    path('create_data_analyst_account/', views.create_data_analyst_account, name='create_data_analyst_account'),

    #view user accounts
    path('view_user_accounts/', views.view_user_accounts, name='view_user_accounts'),
    path('view_businessman_accounts/', views.view_businessman_accounts, name='view_businessman_accounts'),
    path('view_content_creator_accounts/', views.view_content_creator_accounts, name='view_content_creator_accounts'),
    path('view_data_analyst_accounts/', views.view_data_analyst_accounts, name='view_data_analyst_accounts'),

    path('categories/', views.category_list, name='category_list'),
    path('create_category/', views.create_category, name='create_category'),
    path('category/update/<int:category_id>/', views.update_category, name='update_category'),

    path('update_user_account/<int:user_id>/', views.update_user_account, name='update_user_account'),

    #neo4j



    #extract data


    #auto-preprocee & manual-preprocess
    path('auto_preprocess/', views.auto_preprocess, name='auto_preprocess'),
    path('manual_preprocess/', views.manual_preprocess, name='manual_preprocess'),

    #download csv
    path('download_csv/', views.download_csv, name='download_csv'),
    #upload csv
    path('upload_csv_or_xlsx/', views.upload_csv_or_xlsx, name= 'upload_csv_or_xlsx'),
    #visualization
    path('create_or_view_visualization/', views.create_or_view_visualization, name = 'create_or_view_visualization'),
    path('dash/', views.dashboard_view, name= 'dash'),

    # machine - learning
    path('test_predictive_models/', views.test_predictive_models, name= 'test_predictive_models'),


    path('scrape/', views.scrape_profile, name='scrape_profile'),
    path('login/', views.login_instagram, name='login_instagram'),

    # visual charts
    path('upload_and_view_charts/', views.upload_and_view_charts, name='upload_and_view_charts'),
]

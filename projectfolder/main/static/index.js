$(document).ready(function () {
    // Handle the Edit button click event
    $('.edit-btn').click(function () {
        const categoryId = $(this).data('id');
        const categoryName = $(this).data('name');
        const categoryDescription = $(this).data('description');

        // Populate the modal fields with the data
        $('#category-id').val(categoryId);
        $('#category-name').val(categoryName);
        $('#category-description').val(categoryDescription);

        // Show the modal
        $('#edit-modal').show();
    });

    // Handle the form submission via AJAX
    $('#edit-category-form').submit(function (e) {
        e.preventDefault(); // Prevent default form submission

        const categoryId = $('#category-id').val();
        const categoryName = $('#category-name').val();
        const categoryDescription = $('#category-description').val();

        $.ajax({
            url: `/category/update/${categoryId}/`,  // Make sure this URL matches your Django URL pattern
            type: 'POST',
            dataType: 'json', // Expecting JSON response
            data: {
                csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
                name: categoryName,
                description: categoryDescription,
            },
            success: function (response) {
                if (response.status === 'success') {
                    // Update the category in the list without refreshing the page
                    const categoryItem = $(`#category-${categoryId}`);
                    categoryItem.find('.category-name').text(response.name);
                    categoryItem.find('.category-description').text(response.description);

                    // Close the modal
                    $('#edit-modal').hide();
                } else {
                    alert('Failed to update category: ' + response.message);
                }
            },
            error: function (xhr, status, error) {
                console.error('Error:', error);
                alert('An error occurred while updating the category.');
            }
        });
    });

    // Handle modal close button
    $('#close-modal').click(function () {
        $('#edit-modal').hide();
    });
});






$(document).ready(function () {
    $('.edit-btn').on('click',function () {
        // get the data from the clicked button
        const userId = $(this).data('id');
        const username = $(this).data('username');
        const email = $(this).data('email');
        const role = $(this).data('role');

        $('#user-id').val(userId);
        $('#username').val(username);
        $('#email').val(email);
        $('#role').val(role);

        $('#edit-modal').show();

       
        });

        $('#close-modal').on('click' ,function () {
            $('#edit-modal').hide();
        });

    $('#edit-user-form').submit(function (event){
        event.preventDefault();

        const userId = $('#user-id').val();
        const username = $('#username').val();
        const email = $('#email').val();
        const role = $('#role').val();

        $.ajax( {
            url : `/update_user_account/${userId}/`,
            method : 'POST',
            data : {
                'user_id' : userId,
                'username' : username,
                'email' : email,
                'role' : role,
                'csrfmiddlewaretoken' : $('input[name=csrfmiddlewaretoken]').val(),
            },
            success : function (response) {
                if (response.status === 'success') {
                    alert('User updated successfully.');
                    location.reload();

                }
                else {
                    alert ('Error updating account');
                }
            },
            error : function () {
                alert ('Error updating account');
            }
        })
    })
        
    });





    

    document.getElementById('create-visualization').addEventListener('click', (e) => {
        e.preventDefault();
        const networkType = document.getElementById('network-type').value;
        const layoutStyle = document.getElementById('layout-style').value;


    //show a loading message
    const messageArea =  document.getElementById('message-area');
    messageArea.textContent = 'Generating visualization...';

    //Send the request to the server
    fetch('/create_visualization/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        },
        body: JSON.stringify({
            networkType: document.getElementById('network-type').value,
            layoutStyle: document.getElementById('layout-style').value,
        }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('visualization-placeholder').innerHTML = `<img src="${data.image}" alt="Network Visualization" style="max-width: 100%; height: auto;">`;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            alert(`Error: ${error.message}`);
        });
    
})    
    
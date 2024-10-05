// Add an event listener to the form to trigger when the form is submitted
document.getElementById('upload-form').addEventListener('submit', function(event) {
    // Prevent the form from submitting immediately to allow showing the loading symbol
    event.preventDefault();

    // Show loading symbol
    document.getElementById('loading').style.display = 'block';

    // Hide the form after submission to prevent further interaction
    this.style.display = 'none';

    // Submit the form after showing the loading symbol
    this.submit();
});


var typingContainer = document.querySelector('.typing');

function typeString(string) {
    return new Promise(resolve => {
        var i = 0;
        var typingInterval = setInterval(function () {
            typingContainer.textContent += string[i];
            i++;
            if (i === string.length) {
                clearInterval(typingInterval); 
                typingContainer.innerHTML += '<br>'; 
                resolve(); 
            }
        }, 50); 
    });
}

// Event listener for file input to display the selected file name
document.getElementById('file-upload').addEventListener('change', function() {
    var fileName = this.files[0].name;  // Get the name of the uploaded file
    document.getElementById('file-name').textContent = `Uploaded Image: ${fileName}`;  // Display the file name in the placeholder
});

// Remove duplicate form submission listener (already handled above)
async function typeAllStrings() {
    for (const string of strings) {
        await typeString(string); 
    }
}

// Start typing effect on page load
typeAllStrings();  

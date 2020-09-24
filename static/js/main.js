$(document).ready(function () {

    // hide the image viewer section
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
 
    
    $("#imageUpload").change(function() {
        console.log("this happens")
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    })
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('This is a:  ' + data);
                console.log('Success!');
            },
        });
    });

});

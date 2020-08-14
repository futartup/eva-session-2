const uploadAndClassifyImage = (id, url, result_id) => {
    var fileInput = document.getElementById(id).files;
    if (!fileInput.length) {
        alert("Please choose a file to upload first");
    }

    var file = fileInput[0];
    var filename = file.name;

    var formData = new FormData();
    formData.append(filename, file)

    console.log(filename);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: url,
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function(response){
        console.log(response);
        if (id == 'facealignment34FileUpload')
        {
            var image = btoa(response);
            $("#uploaded_image").attr("src", 'data:image/jpeg;base64,' + image);
        }
        else
        {
            document.getElementById(result_id).textContent = response;
        }
    })
    .fail(function(){
        alert("There is an error")
    });
};

// Urls
const resnet_url = "https://ua5wxk6.execute-api.ap-south-1.amazonaws.com/dev/classify"
const mobilenet_url = "https://xywqwgo56g.execute-api.ap-south-1.amazonaws.com/dev/classify"
const facealigned_url = ""

// Resnet example
$('#btnResNetUpload').click(function(){
    uploadAndClassifyImage('resnet34FileUpload', resnet_url, 'result')
});

// Mobilenet example
$('#btnMobileNetUpload').click(function(){
    uploadAndClassifyImage('mobilenet34FileUpload', mobilenet_url, 'result')
});


// Mobilenet example
$('#btnFaceAlignmentUpload').click(function(){
    uploadAndClassifyImage('facealignment34FileUpload', facealigned_url, 'result')
});

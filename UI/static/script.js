function predictSentiment() {
    $('#resultContainer').empty();
    $('#loading').show();
    
    var comment = $('#comment').val();
    if (comment.trim() === '') {
        $('#loading').hide();
        $('#resultContainer').html('<h2>Emotion not defined</h2>');
        return;
    }
    $.ajax({
        type: 'POST',
        url: '/predict_sentiment',
        data: { comment: comment },
        success: function(result) {
            $('#loading').hide();
            $('#resultContainer').html('<p>Emotion:&nbsp;</p><h2>' + result + '</h2>');
        },
        complete: function() {
            $('#loading').hide();
        }
    });
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        predictSentiment();
    }
}


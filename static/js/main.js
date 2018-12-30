/* global $ */
function getInput(){
    
    var x_input = document.getElementById('text_input').value;
    $.ajax({
        url: '/api/model', 
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(x_input),
        success: (data) => {
            for (let i = 0; i < 7; i++) {
                var max = 0;
                var max_index = 0;
                for (let j = 0; j < 10; j++) {
                    var value = Math.round(data.results[i][j] * 1000);
                    if (value > max) {
                        max = value;
                        max_index = j;
                    }
                    var digits = String(value).length;
                    for (var k = 0; k < 3 - digits; k++) {
                        value = '0' + value;
                    }
                    var text = '0.' + value;
                    if (value > 999) {
                        text = '1.000';
                    }
                    $('#output tr').eq(j + 1).find('td').eq(i).text(text);
                }
                for (let j = 0; j < 10; j++) {
                    if (j == max_index) {
                        /* $('#output tr').eq(j + 1).find('td').eq(i).addClass('success'); */
                        $('#output tr').eq(j + 1).find('td').eq(i).css('background-color', 'blue');
                    } 
                    else {
                        /* $('#output tr').eq(j + 1).find('td').eq(i).removeClass('success'); */
                        $('#output tr').eq(j + 1).find('td').eq(i).css("background-color", "");
                    }
                }
            }
        }
    });
}

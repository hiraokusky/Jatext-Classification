#Credits to Lin Zhouhan(@hantek) for the complete visualization code
import random, os, numpy, scipy
from codecs import open
def createHTML(result, texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    fileName = "visualization/attention/"+fileName
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    """
    part2 = """
    </body>
    <script>
    """
    part3 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = 0.0;
            for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
                if (i+j < intensity.length && i+j > -1) {
                    intensity[i] += trigram_weights[k][i + j];
                }
            }
            if (i == 0 || i == intensity.length-1) {
                intensity[i] /= 2.0;
            } else {
                intensity[i] /= 3.0;
            }
            if (intensity[i] > max_intensity) {
                max_intensity = intensity[i];
            }
            if (intensity[i] < min_intensity) {
                min_intensity = intensity[i];
            }
        }
        var denominator = max_intensity - min_intensity;
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = (intensity[i] - min_intensity) / denominator;
        }
        if (k%2 == 0) {
            var heat_text = "<p><br><b>" + result_text[k] + "</b><br>";
        } else {
            var heat_text = "<b>" + result_text[k] + "</b><br>";
        }
        var space = "";
        for (var i = 0; i < tokens.length; i++) {
            heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
            if (space == "") {
                space = " ";
            }
        }
        //heat_text += "<p>";
        document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    resultString = "var result_text = ["
    n = 0
    for r in result:
        if n > 0:
            resultString += ","
        resultString += "\"" + r[0] + " -> 1:" + r[2][0] + " 2:" + r[2][1] + " 3:" + r[2][2] + " 4:" + r[2][3] + " 5:" + r[2][4] + "\""
        n += 1
    resultString += "];\n"
    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write('<h3>' + fileName + '</h3>')
    fOut.write(part2)
    fOut.write(resultString)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part3)
    fOut.close()
  
    return

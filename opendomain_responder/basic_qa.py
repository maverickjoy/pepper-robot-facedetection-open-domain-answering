import re
import json
import requests
import mechanize
from bs4 import BeautifulSoup
from geo_locator import GeoLocator

# INITIALISING AUTO ANSWER BOT
br = mechanize.Browser()
br.set_handle_equiv(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)
br.set_handle_refresh(False)
br.addheaders = [('User-agent', 'Firefox')]


climate_list = ['climate', 'temp', 'temperature',
                'weather', 'hot', 'cold', 'humidity', 'rainy', 'rain']

# Weather API Initialisation
# base_url variable to store url
base_url = "http://api.openweathermap.org/data/2.5/weather?"
api_key = "AP_KEY_GET_FROM_OPENWEATHERMAP_FOR_FREE"


def answerQues(ques):
    ans = ""
    try:
        net_detect = 0
        response = br.open('https://www.google.co.in')
        net_detect = 1
        br.select_form(nr=0)
        br.form['q'] = ques
        br.submit()
        src_code = br.response().read()

        def _checkWeatherQuestion(question):
            question = re.sub(r'[?|$|.|!]', r'', question)
            question = re.sub(r'[^a-zA-Z0-9 ]', r'', question)
            for ele in climate_list:
                if ele in question.lower():
                    return True
            return False

        def _findTemperature(question):
            forecast = "Sorry I'm presently unable to get weather information at the moment"

            places = GeoLocator(question)
            city_name = 'Pune'  # default
            if len(places.cities) > 0:
                city_name = places.cities[0]

            complete_url = base_url + "appid=" + api_key + "&q=" + city_name
            response = requests.get(complete_url)
            x = response.json()

            if x["cod"] != "404":
                y = x["main"]
                current_temperature = y["temp"]
                current_pressure = y["pressure"]
                current_humidiy = y["humidity"]
                z = x["weather"]
                weather_description = z[0]["description"]

                forecast = "Presently in {} it is {} with {} degree Centigrade and humidity percentage is {}".format(
                    city_name, str(weather_description), str(current_temperature - 273), str(current_humidiy))

            return forecast

        def _whois():
            answer = re.search('<span>(.*)', src_code)
            answer = answer.group()[:400]
            if 'wiki' in answer:
                answer = re.search('<span>(.*)<a', answer).group(1)
            else:
                answer = re.search(
                    '<span>(.*)</span></div></div><div', answer).group(1)

            return answer

        def _whatis():
            spg = 1
            answer = re.search('"_sPg">(.*)', src_code)
            if answer == None:
                answer = re.search('<ol><li>(.*)', src_code)
                spg = 0
            if answer == None:
                return _whois()
            else:
                answer = answer.group()[:400]
                if '<b>' in answer:
                    answer = answer.replace('<b>', '')
                    answer = answer.replace('</b>', '')
                if spg:
                    answer = re.search(
                        '"_sPg">(.*)</div></div><div', answer).group(1)
                else:
                    answer = re.search('<ol><li>(.*)</li>', answer).group(1)

                return answer

        if _checkWeatherQuestion(ques):
            ans = _findTemperature(ques)
        elif 'who' in ques:
            ans = _whois()
        else:
            ans = _whatis()

    except Exception as err:
        print "Cannot find answer segment : ", err

    ans = str(BeautifulSoup(ans, "html.parser").text)
    return ans


if __name__ == "__main__":
    while True:
        print "Please enter your question"
        ques = raw_input('Ques : ')
        ans = answerQues(ques)
        print "> ", ans

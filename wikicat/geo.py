"""Bare country and continent names. Suggesting "India" or "Kenya" on its own
adds little for editors, so these categories can be hidden from suggestions;
specific place categories such as "Beaches in Goa" are kept."""
from __future__ import annotations

COUNTRIES = """
Afghanistan|Albania|Algeria|Andorra|Angola|Antigua and Barbuda|Argentina|Armenia|Australia|Austria|Azerbaijan|
Bahamas|The Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia and Herzegovina|
Botswana|Brazil|Brunei|Bulgaria|Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Cape Verde|Central African Republic|
Chad|Chile|China|People's Republic of China|Colombia|Comoros|Democratic Republic of the Congo|Republic of the Congo|
Costa Rica|Croatia|Cuba|Cyprus|Czech Republic|Czechia|Denmark|Djibouti|Dominica|Dominican Republic|East Timor|
Timor-Leste|Ecuador|Egypt|El Salvador|Equatorial Guinea|Eritrea|Estonia|Eswatini|Swaziland|Ethiopia|Fiji|Finland|
France|Gabon|The Gambia|Gambia|Georgia (country)|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guinea-Bissau|Guyana|
Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Republic of Ireland|Israel|Italy|Ivory Coast|
Côte d'Ivoire|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kiribati|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|
Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Marshall Islands|
Mauritania|Mauritius|Mexico|Federated States of Micronesia|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|
Myanmar|Namibia|Nauru|Nepal|Netherlands|the Netherlands|New Zealand|Nicaragua|Niger|Nigeria|North Korea|
North Macedonia|Norway|Oman|Pakistan|Palau|Palestine|State of Palestine|Panama|Papua New Guinea|Paraguay|Peru|
Philippines|the Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Saint Kitts and Nevis|Saint Lucia|
Saint Vincent and the Grenadines|Samoa|San Marino|São Tomé and Príncipe|Saudi Arabia|Senegal|Serbia|Seychelles|
Sierra Leone|Singapore|Slovakia|Slovenia|Solomon Islands|Somalia|South Africa|South Korea|South Sudan|Spain|
Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Tonga|
Trinidad and Tobago|Tunisia|Turkey|Türkiye|Turkmenistan|Tuvalu|Uganda|Ukraine|United Arab Emirates|
United Kingdom|the United Kingdom|United States|United States of America|the United States|Uruguay|Uzbekistan|
Vanuatu|Vatican City|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe|
Africa|Asia|Europe|North America|South America|Central America|Latin America|Oceania|Antarctica|Middle East|
Caribbean|the Caribbean|World|Earth
"""

COUNTRY_NAMES = frozenset(n.strip().lower() for n in COUNTRIES.replace("\n", "").split("|") if n.strip())


def is_country(name: str) -> bool:
    return name.strip().lower() in COUNTRY_NAMES

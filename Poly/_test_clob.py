import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

host = "https://clob.polymarket.com"
chain_id = 137 # polygon

api_key = "019d6439-ffd5-7b5c-a090-44c3c8eac5a0"
secret = "vdnzQ1ec_rl1F1q_SNDR8nMwwSOwBjf6ek7FCiIZ6H8="
passphrase = "a60251629c5e442d0ed2bdd5dad1e368cebc273be902132d2dfd15f112114a20"
funder = "0x18a413fc7ac37063d55a3e9fab8630fd5160d80c"

creds = ApiCreds(api_key=api_key, api_secret=secret, api_passphrase=passphrase)
# Note: we might not need a private key if we use session credentials (ApiCreds).
try:
    client = ClobClient(host, key="", chain_id=chain_id, creds=creds, signature_type=2, funder=funder)
    # let's try to get auth token
    print("Trying to auth / token...")
    # we don't necessarily need the token for just querying but let's check ok
    res = client.get_ok()
    print("L2 OK:", res)
    # Check if we can create a client and check orderbook
    print("Client initialized fully.")
except Exception as e:
    print("Error:", e)

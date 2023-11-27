#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep

BASE_URL = "https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/IPR017736/?page_size=200&extra_fields=sequence"

HEADER_SEPARATOR = "|"
LINE_LENGTH = 80

def output_list():
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = BASE_URL
  last_page = False

  
  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      
      entries = None
      if ("entry_subset" in item):
        entries = item["entry_subset"]
      elif ("entries" in item):
        entries = item["entries"]
      
      if entries is not None:
        entries_header = "-".join(
          [entry["accession"] + "(" + ";".join(
            [
              ",".join(
                [ str(fragment["start"]) + "..." + str(fragment["end"]) 
                  for fragment in locations["fragments"]]
              ) for locations in entry["entry_protein_locations"]
            ]
          ) + ")" for entry in entries]
        )
        sys.stdout.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR
                          + entries_header + HEADER_SEPARATOR
                          + item["metadata"]["name"] + "\n")
      else:
        sys.stdout.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR + item["metadata"]["name"] + "\n")

      seq = item["extra_fields"]["sequence"]
      fastaSeqFragments = [seq[0+i:LINE_LENGTH+i] for i in range(0, len(seq), LINE_LENGTH)]
      for fastaSeqFragment in fastaSeqFragments:
        sys.stdout.write(fastaSeqFragment + "\n")
      
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(0.1)

if __name__ == "__main__":
  output_list()


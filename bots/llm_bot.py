import re
import urllib.request
import urllib.error
import json
import random


OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3.1:8b"

RULES = """
You are playing Liar's Dice. Rules:
- Each player starts with 5 dice and rolls them secretly each round.
- Players take turns making bids. A bid is (quantity, face) meaning "there are at least QUANTITY dice showing FACE across all players combined."
- Face 1 is a wildcard and counts toward any face bid.
- Each new bid must be strictly higher: either a higher quantity, or same quantity with a higher face value.
- Instead of bidding, you may call 'bluff' (you think the previous bid is false) or 'spot on' (you think the count is exactly right).
- If you call 'bluff' and the actual count is less than the bid -> previous player loses a die. Otherwise you lose a die.
- If you call 'spot on' and the count exactly matches -> all other players lose a die. Otherwise you lose a die.
- You CANNOT call 'bluff' or 'spot on' if you are the first bidder in a round.
- A player with 0 dice is eliminated. Last player standing wins.
""".strip()


def _allBids(prevBet, totalDice):
    bids = []
    for n in range(1, totalDice + 1):
        for v in range(2, 7):
            if prevBet is None:
                bids.append((n, v))
            else:
                pn, pv = prevBet
                if n > pn or (n == pn and v > pv):
                    bids.append((n, v))
    return bids


def _getPlayers(state):
    if 'players' in state:
        return state['players']
    active = state.get('active_players', [])
    counts = state.get('dice_counts', [])
    return [{'id': pid, 'num_die': counts[pid]} for pid in active]


def _buildPrompt(state):
    myId    = state['your_id']
    myRolls = sorted(state['your_rolls'])
    prevBet = state.get('previous_bet') or state.get('previous_bid')
    players = _getPlayers(state)
    totalDice = sum(p['num_die'] for p in players if p['num_die'] > 0)
    isFirst = (prevBet is None)

    lines = [RULES, ""]
    lines.append(f"You are Player {myId}.")
    lines.append(f"Your dice: {myRolls}")
    lines.append(f"Total dice in play: {totalDice}")
    lines.append(f"Players and their dice counts: { {p['id']: p['num_die'] for p in players if p['num_die'] > 0} }")

    if prevBet:
        lines.append(f"Previous bid: {prevBet[0]} x {prevBet[1]}")
    else:
        lines.append("You are the first bidder this round.")

    lines.append("")
    if isFirst:
        lines.append("You must place a bid. Respond with ONLY: BID <quantity> <face>")
        lines.append("Example: BID 3 4")
    else:
        lines.append("Choose one action. Respond with ONLY one of:")
        lines.append("  BID <quantity> <face>   (e.g. BID 4 3)")
        lines.append("  BLUFF")
        lines.append("  SPOT ON")

    return "\n".join(lines)


def _queryOllama(prompt):
    payload = json.dumps({
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 20}
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip().upper()
    except (urllib.error.URLError, Exception):
        return None


def _parseResponse(text, prevBet, totalDice, isFirst):
    if text is None:
        return None

    if not isFirst:
        if re.search(r'\bBLUFF\b', text):
            return 'bluff'
        if re.search(r'\bSPOT\s+ON\b', text):
            return 'spot on'

    m = re.search(r'BID\s+(\d+)\s+(\d+)', text)
    if m:
        qty, face = int(m.group(1)), int(m.group(2))
        if 2 <= face <= 6 and 1 <= qty <= totalDice:
            if prevBet is None:
                return (qty, face)
            pn, pv = prevBet
            if qty > pn or (qty == pn and face > pv):
                return (qty, face)

    return None


def _fallback(prevBet, totalDice, isFirst):
    if not isFirst and prevBet:
        pn, pv = prevBet
        ratio = pn / max(1, totalDice)
        if ratio > 0.7 and random.random() < 0.8:
            return 'bluff'

    bids = _allBids(prevBet, totalDice)
    return random.choice(bids[:3]) if bids else 'bluff'


def llm_bot(state):
    myId    = state['your_id']
    prevBet = state.get('previous_bet') or state.get('previous_bid')
    players = _getPlayers(state)
    totalDice = sum(p['num_die'] for p in players if p['num_die'] > 0)
    isFirst = (prevBet is None)

    if state.get('current_id') is not None and state['current_id'] != myId:
        return None

    prompt   = _buildPrompt(state)
    rawText  = _queryOllama(prompt)
    response = _parseResponse(rawText, prevBet, totalDice, isFirst)

    if response is None:
        response = _fallback(prevBet, totalDice, isFirst)

    legalActs = state.get('legal_actions')
    legalBids = state.get('legal_bids')
    if legalActs and legalBids:
        if response == 'bluff' and 'liar' in legalBids:
            return legalActs[legalBids.index('liar')]
        elif response == 'spot on' and 'spot_on' in legalBids:
            return legalActs[legalBids.index('spot_on')]
        elif response in legalBids:
            return legalActs[legalBids.index(response)]
        for a, b in zip(legalActs, legalBids):
            if b not in ('liar', 'spot_on'):
                return a

    return response

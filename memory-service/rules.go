package main

// Pure memory rules. No I/O here, so they are unit-testable and mirror skillforge/memory/store.py.

const (
	rrfK             = 60
	minSimilarity    = 0.15
	deprecateMinUses = 5
	deprecateBelow   = 0.5
)

// Fact action names returned to clients.
const (
	ActCreated    = "created"
	ActReinforced = "reinforced"
	ActSuperseded = "superseded"
	ActHistory    = "history"
	ActRejected   = "rejected"
)

type factRec struct {
	Object     string
	ValidFrom  float64
	Confidence float64
	ExpiresAt  *float64
}

// decideFact applies the write rules for a fact keyed on (subject, predicate).
// cur is the live (active, unexpired) incumbent or nil.
func decideFact(cur *factRec, in factRec, now float64) string {
	if cur == nil {
		if in.ExpiresAt != nil && *in.ExpiresAt <= now {
			return ActHistory // arrived already expired
		}
		return ActCreated
	}
	if cur.Object == in.Object {
		return ActReinforced
	}
	if in.ValidFrom > cur.ValidFrom || (in.ValidFrom == cur.ValidFrom && in.Confidence > cur.Confidence) {
		return ActSuperseded
	}
	if in.ValidFrom < cur.ValidFrom {
		return ActHistory // late-arriving stale information never wins
	}
	return ActRejected
}

// shouldDeprecate: enough evidence and a success rate below the threshold.
func shouldDeprecate(uses, successes int) bool {
	return uses >= deprecateMinUses && float64(successes)/float64(uses) < deprecateBelow
}

// rrf fuses ranked id lists with reciprocal rank fusion and returns ids by descending score.
func rrf(lists ...[]int64) ([]int64, map[int64]float64) {
	score := map[int64]float64{}
	var order []int64
	for _, l := range lists {
		for r, id := range l {
			if _, ok := score[id]; !ok {
				order = append(order, id)
			}
			score[id] += 1.0 / float64(rrfK+r+1)
		}
	}
	// insertion sort keeps this dependency-free and stable for small k
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && score[order[j]] > score[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	return order, score
}

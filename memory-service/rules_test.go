package main

import "testing"

func f(o string, vf, c float64) factRec { return factRec{Object: o, ValidFrom: vf, Confidence: c} }

func TestDecideFact(t *testing.T) {
	past := 5.0
	cases := []struct {
		name string
		cur  *factRec
		in   factRec
		now  float64
		want string
	}{
		{"create", nil, f("a", 1, .8), 1, ActCreated},
		{"already expired", nil, factRec{Object: "a", ValidFrom: 0, ExpiresAt: &past}, 10, ActHistory},
		{"reinforce", &factRec{Object: "a", ValidFrom: 1}, f("a", 2, .8), 2, ActReinforced},
		{"newer supersedes", &factRec{Object: "a", ValidFrom: 1, Confidence: .8}, f("b", 2, .8), 2, ActSuperseded},
		{"tie higher confidence", &factRec{Object: "a", ValidFrom: 1, Confidence: .5}, f("b", 1, .9), 2, ActSuperseded},
		{"older is history", &factRec{Object: "a", ValidFrom: 10, Confidence: .8}, f("b", 5, .99), 11, ActHistory},
		{"tie lower confidence rejected", &factRec{Object: "a", ValidFrom: 1, Confidence: .8}, f("b", 1, .5), 2, ActRejected},
	}
	for _, c := range cases {
		if got := decideFact(c.cur, c.in, c.now); got != c.want {
			t.Errorf("%s: got %s want %s", c.name, got, c.want)
		}
	}
}

func TestShouldDeprecate(t *testing.T) {
	if shouldDeprecate(4, 0) {
		t.Fatal("needs min uses")
	}
	if !shouldDeprecate(5, 2) {
		t.Fatal("2/5 < 50% must deprecate")
	}
	if shouldDeprecate(6, 3) {
		t.Fatal("exactly 50% is kept")
	}
}

func TestRRFPrefersAgreement(t *testing.T) {
	order, _ := rrf([]int64{1, 2, 3}, []int64{3, 4, 1})
	if order[0] != 1 && order[0] != 3 {
		t.Fatalf("items ranked by both lists must lead: %v", order)
	}
	if len(order) != 4 {
		t.Fatalf("union expected: %v", order)
	}
}

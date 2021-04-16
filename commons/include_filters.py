from __future__ import annotations
from typing import Any, Type, Union, Literal, List, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .thread_stats import ThreadStats, Counts


class IncludeRule(ABC):

    @classmethod
    def build(cls, builder: IncludeRuleBuilder, args: Any) -> IncludeRule:
        if isinstance(args, dict):
            return cls(**args)
        if isinstance(args, list):
            return cls(*args)
        raise ValueError(args)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def check(self, thread: ThreadStats, ranking: int, counts: Counts, all_threads: List[ThreadStats]) -> bool:
        raise NotImplementedError()


@dataclass(frozen=True)
class IncludeRuleBuilder:

    rule_map: Dict[str, Type[IncludeRule]] = field(default_factory=dict)

    def register_rule(self, name: str, rule_class: Type[IncludeRule]):
        assert(name not in self.rule_map)
        self.rule_map[name] = rule_class

    def build(self, root_rule_obj: Dict[str, Any]) -> IncludeRule:
        if len(root_rule_obj) != 1:
            raise ValueError(root_rule_obj)
        [name, args] = next(iter(root_rule_obj.items()))
        return self.rule_map[name].build(self, args)


include_rule_builder = IncludeRuleBuilder()


@dataclass(frozen=True)
class IncludeRuleRanking(IncludeRule):

    verb: Union[Literal['<='], Literal['=='], Literal['>='],
                Literal['<'], Literal['>']]
    value: Union[int, Literal['@q1'], Literal['@q2'], Literal['@q3']]

    def __post_init__(self):
        if self.verb not in ['<=', '==', '>=', '<', '>']:
            raise KeyError(self.verb)
        if self.verb != '<=':
            raise NotImplementedError(self.verb)

        if not isinstance(self.value, int):
            if self.value not in ['@q1', '@q2', '@q3']:
                raise KeyError(self.value)

    def __str__(self):
        if isinstance(self.value, int):
            return f"前 {self.value} 位"
        if self.value.startswith('@q'):
            qn = int(self.value[2:])
            if qn == '1':
                percentage = '75%'
            elif qn == '2':
                percentage = '50%'
            elif qn == '3':
                percentage = '25%'
            else:
                assert(False)
            return f"前 {percentage}"
        assert(False)

    def check(self, thread: ThreadStats, ranking: int, counts: Counts, all_threads: List[ThreadStats]):
        if isinstance(self.value, int):
            if self.verb == '<=':
                if self.value >= len(all_threads):
                    return True
                return thread.increased_character_count >= all_threads[self.value].increased_character_count
            assert(False)

        if self.value.startswith('@q'):
            qn = int(self.value[2:])
            qn_new_responses = counts.thread_new_post_quartiles[qn-1]
            if self.verb == '<=':
                return thread.increased_response_count >= qn_new_responses
            assert(False)

        assert(False)


include_rule_builder.register_rule('ranking', IncludeRuleRanking)


@dataclass(frozen=True)
class IncludeRuleField(IncludeRule):

    field_name: str
    verb: Union[Literal['<='], Literal['=='], Literal['>='],
                Literal['<'], Literal['>']]
    value: Any

    def __post_init__(self):
        if self.verb not in ['<=', '==', '>=', '<', '>']:
            raise KeyError(self.verb)
        if self.verb != '>=':
            raise NotImplementedError(self.verb)
        if self.field_name not in [
            'increased_response_count',
            'increased_character_count',
        ]:
            raise NotImplementedError(self.field_name)

    def __str__(self):
        if self.field_name == 'increased_response_count':
            ret = f"新增回应≥{self.value}"
            if self.value % 19 == 0:
                ret += f"（满{self.value // 19}页）"
            return ret
        if self.field_name == 'increased_character_count':
            return f"新增文本≥{self.value/1000:.2f}K"
        assert(False)

    def check(self, thread: ThreadStats, ranking: int, counts: Counts, all_threads: List[ThreadStats]):
        if self.verb == '>=':
            return getattr(thread, self.field_name) >= self.value
        assert(False)


include_rule_builder.register_rule('field', IncludeRuleField)


@dataclass(frozen=True)
class IncludeRuleCombinator(IncludeRule):

    children: List[IncludeRule]

    @classmethod
    def build(cls, builder: IncludeRuleBuilder, args: Any) -> IncludeRule:
        if isinstance(args, list):
            args = map(lambda r: builder.build(r), args)
            return cls(list(args))
        raise ValueError(args)

    def __post_init__(self):
        for child in self.children:
            assert(isinstance(child, IncludeRule))

    @property
    def _has_child_combinators(self):
        return len(list(filter(lambda c: isinstance(c, IncludeRuleCombinator), self.children)))


class IncludeRuleAll(IncludeRuleCombinator):

    def check(self, thread: ThreadStats, ranking: int, counts: Counts, all_threads: List[ThreadStats]):
        for child in children:
            if not child.check(thread, ranking, counts, all_threads):
                return False
        return True

    def __str__(self):
        ret = ' 且 '.join(list(map(lambda c: str(c), self.children)))
        if self._has_child_combinators:
            ret = "（" + ret + "）"
        return ret


include_rule_builder.register_rule('all', IncludeRuleAll)


class IncludeRuleAny(IncludeRuleCombinator):

    def check(self, thread: ThreadStats, ranking: int, counts: Counts, all_threads: List[ThreadStats]):
        for child in self.children:
            if child.check(thread, ranking, counts, all_threads):
                return True
        return False

    def __str__(self):
        ret = ' 或 '.join(list(map(lambda c: str(c), self.children)))
        if self._has_child_combinators:
            ret = "（" + ret + "）"
        return ret


include_rule_builder.register_rule('any', IncludeRuleAny)

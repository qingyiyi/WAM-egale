"""Import only after common.setup has bound the original mesh_helper."""
from runtime import Runtime
import tree_api
class EvalRuntime(Runtime):
 def execute_k2_b2_route(self,tree,scores,route,**kwargs):
  result=tree_api.execute_k2_b2_route(tree,scores,route,**kwargs)
  self.tree_records.append(dict(step=self.counters.get('committed_scheduler_update_count',0),depth=result.accepted_depth,
      first=list(scores.first_costs),second=[list(v) for v in scores.second_costs],parent=route.parent,child=route.child))
  return result

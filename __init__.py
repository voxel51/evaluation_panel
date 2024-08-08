import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.brain as fob
from fiftyone.brain import Similarity
import fiftyone as fo
import fiftyone.zoo.models as fozm
import fiftyone.core.patches as fop
from fiftyone import ViewField as F
import fiftyone.core.fields as fof


class EvaluationPanel(foo.Panel):

    @property
    def config(self):
        return foo.PanelOperatorConfig(
            name="evaluation_panel",
            label="evaluation_panel"
        )
    
    # def on_change_view(self, ctx):
    #     print("View changed")
    
    def on_load(self, ctx):
        print("Panel loaded")


    
    def render(self, ctx):
        panel = types.Object()
        
        # Define main stack
        stack = panel.v_stack("my_stack", align_x="center", gap=2)

        stack.md("""
            ### Evaluate Your Models
        """,
        name="md1")
        
        # Add operator buttons
        stack.btn("evaluate_model", label="Evaluate Model", on_click=self.evaluate_model)
        stack.btn("compare_models", label="compare_models", on_click=self.compare_models)



        # Create the eval key options for the menus
        eval_keys = types.Choices()
        for key in ctx.dataset.list_evaluations():
            eval_keys.add_choice(key, label=key)

        menu = stack.menu('menu', variant="square", width=100, align_y="center")
        actions = menu.btn_group('actions')

        # Add Eval Key Menu
        actions.enum('eval_key', label="Evaluation key", values=eval_keys.values(),  view=types.View(space=3), on_change=self.on_change_config,)

        compare_keys = eval_keys.values()

        # Check to see if state initialized the stack
        if ctx.panel.get_state("my_stack") != None:
            eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
            if eval_key != None:

                # Remove active eval key from the compare list
                compare_keys = eval_keys.values()
                index = compare_keys.index(eval_key )
                compare_keys.pop(index)

                # Find the type of eval
                current_eval = ctx.dataset.get_evaluation_info(eval_key)
                eval_type = current_eval.config.type

            
        # Add Compare Key menu
        actions.enum('compare_key', label="Compare key", values=compare_keys,  view=types.View(space=3), on_change=self.on_change_config,)

        # Define Tab View for Eval Results vs Info
        tabs = types.TabsView()
        tabs.add_choice("results", label="Evaluation Results")
        tabs.add_choice("info", label="Evaluation Info")

        
        stack.str("tabs", view=tabs, on_change=self.on_path_change)

        # Define the paths for tabs
        if ctx.panel.get_state("my_stack") != None:
            eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
            if eval_key != None:
                current_tab = ctx.panel.get_state("my_stack.tabs")

                current_eval = ctx.dataset.get_evaluation_info(eval_key)
                if current_tab == 'results':

                    current_eval = ctx.dataset.get_evaluation_info(eval_key).serialize()
                    _eval_results(ctx, stack, current_eval)
                    
                    
                elif current_tab== 'info':
                    if current_eval:
                        _eval_info(ctx, stack, current_eval) 
    
                else:
                    stack.md("# The third tab")

        
        
        

        return types.Property(
            panel,
            view=types.GridView(
                height=100,
                width=100,
                align_x="center",
                align_y="center",
                componentsProps={"container": {"sx": {"position": "relative"}}},
            ),
        )

    def on_click_install(self, ctx):
        ctx.ops.track_event("try_visdrone_panel_install")

    def on_click_schedule_demo(self, ctx):
        ctx.ops.track_event("try_visdrone_panel_schedule_demo")

    def on_plot_one_click(self, ctx):
        ctx.ops.track_event("try_visdrone_plot_one_click")
        ctx.ops.set_active_fields(["ground_truth", "First Model"])

        params = ctx.params["data"]
        if params["x"] == params["y"]:

            # Filter only X/Y labels

            view1 = ctx.dataset.filter_labels("First Model", F("eval_first").is_in("tp"))
            view2 = view1.filter_labels("ground_truth", F("eval_first").is_in("tp"))
            view3 = view2.filter_labels("First Model", F("label").is_in(params["x"]))
            view4 = view3.filter_labels("ground_truth", F("label").is_in(params["x"]))
            final_view = view4.filter_labels("Best Model", F("label").is_in(""), only_matches=False)
        else:
            # Grab FP
            view1 = ctx.dataset.filter_labels("First Model", F("eval_first").is_in("fp"))

            # Grab only samples with label X/Y
            view2 = view1.filter_labels("First Model", F("label").is_in(params["y"]))
            view3 = view2.filter_labels("ground_truth", F("label").is_in(params["x"]))
            final_view = view3.filter_labels("Best Model", F("label").is_in(""), only_matches=False)

        ctx.ops.set_view(final_view)

    def on_plot_two_click(self, ctx):
        ctx.ops.track_event("try_visdrone_plot_two_click")
        ctx.ops.set_active_fields(["ground_truth", "Best Model"])
        params = ctx.params
        if params["x"] == params["y"]:



            # Filter only X/Y labels

            view1 = ctx.dataset.filter_labels("Best Model", F("eval_best").is_in("tp"))
            view2 = view1.filter_labels("ground_truth", F("eval_best").is_in("tp"))
            view3 = view2.filter_labels("Best Model", F("label").is_in(params["x"]))
            view4 = view3.filter_labels("ground_truth", F("label").is_in(params["x"]))
            final_view = view4.filter_labels("First Model", F("label").is_in(""), only_matches=False)
        else:
            # Grab FP
            view1 = ctx.dataset.filter_labels("Best Model", F("eval_best").is_in("fp"))

            # Grab only samples with label X/Y
            view2 = view1.filter_labels("Best Model", F("label").is_in(params["y"]))
            view3 = view2.filter_labels("ground_truth", F("label").is_in(params["x"]))
            final_view = view3.filter_labels("First Model", F("label").is_in(""), only_matches=False)

        ctx.ops.set_view(final_view)
        


    def on_plot_click(self, ctx):
        ctx.ops.track_event("try_visdrone_plot_click")

        type = ctx.panel.state.plot_config.get('plot_type')
        # print('on plot click', ctx.params)
        if type == "histogram":
            min = ctx.params.get("range", [])[0]
            max = ctx.params.get("range", [])[1]
            filter = {}
            filter[ctx.params.get("x_data_source")] = {"$gte": min, "$lte": max}
            ctx.trigger("set_view", dict(view=[
                {
                    "_cls": "fiftyone.core.stages.Match",
                    "kwargs": [
                    [
                        "filter",
                        filter
                    ]
                    ],
                }
            ]))
        elif type == "eval_results":
            print('heatmap x', ctx.params.get("x"))
            print('heatmap y', ctx.params.get("y"))
            set_view_for_confustion_matrix_cell(ctx, "ground_truth", "predictions", ctx.params.get("x"), ctx.params.get("y"))

    def update_plot_data(self, ctx):
        print('update plot data --------')
        print(ctx.params)
        success_result = ctx.params.get('result', {})
        success_results_plot_config = success_result.get('plot_config', None)
        plot_config = success_results_plot_config or ctx.panel.state.plot_config
        data = get_plot_data(ctx.dataset, plot_config)
        ctx.panel.data.plot = data
        if success_results_plot_config:
            ctx.panel.state.plot_config = plot_config

    def on_success(self, ctx):
        print('on success')
        print(ctx.params)
        result = ctx.params.get('result', )
        new_config = result.get('plot_config')
        ctx.panel.state.plot_config = new_config
        self.update_plot_data(ctx)

    def on_click_configure(self, ctx):
        ctx.prompt("@voxel51/try_fiftyone/plotly_plot_operator", params=ctx.panel.state.plot_config, on_success=self.update_plot_data)
        
    def on_click_start(self, ctx):
        ctx.panel.state.welcome = False

    def go_to_next_page(self, ctx):
        if ctx.panel.state.page == 1:
            ctx.panel.state.view = ""
        ctx.panel.state.page = ctx.panel.state.page+1
        

    def go_to_previous_page(self, ctx):
        ctx.panel.state.page = ctx.panel.state.page-1

    def reset_page(self, ctx):
        ctx.panel.state.page = 1
        ctx.panel.state.view = ""

    def evaluate_model(self, ctx):
        ctx.prompt("@voxel51/evaluation/evaluate_model")
        ctx.panel.state.view = "eval"

    def compare_models(self, ctx):
        ctx.prompt("@allenleetc/model_comparison/compute_changes")
        ctx.panel.state.view = "compare"

    def on_path_change(self, ctx):
        #ctx.trigger("reload_samples")

        print("on_change:", ctx.params)

    def on_change_config(self, ctx):
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        current_eval = ctx.dataset.get_evaluation_info(eval_key).serialize()
        eval_type = current_eval["config"]["type"]
        results = ctx.dataset.load_evaluation_results(eval_key)
        report = results.report()
        table_list = []
        for key in report:
            table_list.append(
                {"class": key,
                "precision": report[key]["precision"],
                "recall": report[key]["recall"],
                "f1-score": report[key]["f1-score"],
                "support": report[key]["support"],
                }
            )
        ctx.panel.set_state("my_stack.evaluations", table_list) 

        if eval_type == "detection":
            if current_eval["config"]["compute_mAP"]:
                mAP_list = []
                for i,label in enumerate(results.classes):
                    new_row = {"class": label, "AP": results._classwise_AP[i]}
                    mAP_list.append(new_row)

                new_row = {"class": "All", "AP": results.mAP()}
                mAP_list.append(new_row)
                ctx.panel.set_state("my_stack.mAP_evaluations", mAP_list) 


            

def _eval_results(ctx, stack, current_eval):

    eval_type = current_eval["config"]["type"]
    stack.md("### Evaluation Results")

    # Define table structure
    item_obj = types.Object()
    item_obj.str('class')
    item_obj.float('precision')
    item_obj.float('recall')
    item_obj.float('f1-score')
    item_obj.float('support')

    # Define the table
    table_view = types.TableView() 
    table_view.add_column('class', label="Class")
    table_view.add_column('precision', label="Precision")
    table_view.add_column('recall', label="Recall")
    table_view.add_column('f1-score', label="F1-Score")
    table_view.add_column('support', label="Support")

    # Add table to the stack
    stack.list('evaluations', element_type=item_obj, view=table_view)

    
    if eval_type and eval_type == "detection":

        if current_eval["config"]["compute_mAP"]:
            mAP_obj = types.Object()
            mAP_obj.str('class')
            mAP_obj.float('AP')


            mAP_table_view = types.TableView() 
            mAP_table_view.add_column('class', label="Class")
            mAP_table_view.add_column('AP', label="Average Precision")
            stack.list('mAP_evaluations', element_type=mAP_obj, view=mAP_table_view)
        else:
            stack.md("No mAP copmuted on this eval_key")

def _eval_info(ctx, stack, current_eval):
    eval_info = current_eval.serialize()

    if current_eval is not None:
    

        # Run info
        stack.view(
            "info_header",
            types.Header(label="Run info", divider=True),
        )
        stack.str(
            "info_eval_key",
            label="Eval key",
            default=eval_info["key"],
            view=types.LabelValueView(read_only=True),

        )
        stack.str(
            "info_run_type",
            label="Run type",
            default=eval_info["config"]["cls"],
            view=types.LabelValueView(read_only=True),
        )
        stack.str(
            "info_timestamp",
            label="Creation time",
            default=str(eval_info["timestamp"]),
            view=types.LabelValueView(read_only=True),
        )
        stack.str(
            "info_version",
            label="FiftyOne version",
            default=eval_info["version"],
            view=types.LabelValueView(read_only=True),
        )

        # Config
        stack.view(
            "config_header",
            types.Header(label="Eval config", divider=True),
        )
        if ctx.params.get("config_raw", False):
            stack.obj(
                "config_json",
                default=eval_info["config"],
                view=types.JSONView(),
            )
        else:
            for key, value in eval_info["config"].items():
                if isinstance(value, dict):
                    stack.obj(
                        "config_" + key,
                        label=key,
                        default=value,
                        view=types.JSONView(),
                    )
                else:
                    stack.str(
                        "config_" + key,
                        label=key,
                        default=str(value),
                        view=types.LabelValueView(read_only=True),
                    )

        stack.bool(
            "config_raw",
            label="Show as JSON",
            default=False,
            view=types.SwitchView(),
        )


class EvaluationMethod(object):
    def __init__(self, eval_type, method):
        self.eval_type = eval_type
        self.method = method

    def get_parameters(self, ctx, inputs):
        pass

    def parse_parameters(self, ctx, params):
        pass


class Regression(EvaluationMethod):
    def get_parameters(self, ctx, inputs):
        inputs.float(
            "missing",
            label="Missing",
            description=(
                "A missing value. Any None-valued regressions are given this "
                "value for results purposes"
            ),
        )


class SimpleRegression(Regression):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Simple regression evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#simple-evaluation-default",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        metric_choices = types.DropdownView()
        metric_choices.add_choice("squared_error", label="squared error")
        metric_choices.add_choice("absolute_error", label="absolute error")

        inputs.enum(
            "metric",
            metric_choices.values(),
            required=True,
            label="Metric",
            description=(
                "The error metric to use to populate sample/frame-level error "
                "data"
            ),
            view=metric_choices,
        )


class Classification(EvaluationMethod):
    def get_parameters(self, ctx, inputs):
        _get_classes(ctx, inputs)

        inputs.float(
            "missing",
            label="Missing",
            description=(
                "A missing label string. Any None-valued labels are given "
                "this label for results purposes"
            ),
        )

    def parse_parameters(self, ctx, params):
        classes = params.pop("classes", None)
        if classes is not None:
            params["classes"] = _parse_classes(ctx, classes)


class SimpleClassification(Classification):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Simple classification evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#id4",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)


class TopKClassification(Classification):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Top-k classification evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#top-k-evaluation",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        inputs.int(
            "k",
            label="k",
            description="The top-k value to use when assessing accuracy",
        )


class BinaryClassification(Classification):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Binary classification evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#binary-evaluation",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)


class Detection(EvaluationMethod):
    def get_parameters(self, ctx, inputs):
        _get_classes(ctx, inputs)

        pred_field = ctx.params["pred_field"]
        label_type, _, _ = _get_evaluation_type(ctx.dataset, pred_field)

        inputs.float(
            "missing",
            label="Missing",
            description=(
                "A missing label string. Any unmatched objects are given this "
                "label for results purposes"
            ),
        )

        inputs.float(
            "iou",
            default=0.5,
            required=True,
            label="IOU",
            description=(
                "The intersection-over-union (IoU) threshold to use to "
                "determine matches"
            ),
        )

        if issubclass(label_type, fo.Detections):
            inputs.bool(
                "use_masks",
                default=False,
                label="Use masks",
                description=(
                    "Whether to compute IoUs using the instances masks of the "
                    "provided objects"
                ),
            )

        if issubclass(label_type, fo.Polylines):
            inputs.bool(
                "use_boxes",
                default=False,
                label="Use boxes",
                description=(
                    "Whether to compute IoUs using the bounding boxes of the "
                    "provided polylines rather than their actual geometries"
                ),
            )

        inputs.bool(
            "classwise",
            default=True,
            label="Classwise",
            description=(
                "Whether to only match objects with the same class label "
                "(True) or allow matches between classes (False)"
            ),
        )

        inputs.bool(
            "dynamic",
            default=True,
            label="Dynamic",
            description=(
                "Whether to declare the dynamic object-level attributes that "
                "are populated on the dataset's schema"
            ),
        )

    def parse_parameters(self, ctx, params):
        classes = params.pop("classes", None)
        if classes is not None:
            params["classes"] = _parse_classes(ctx, classes)


class COCODetection(Detection):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="COCO-style evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#coco-style-evaluation-default-spatial",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        _get_iscrowd(ctx, inputs)

        inputs.bool(
            "compute_mAP",
            default=False,
            label="Compute mAP",
            description=(
                "Whether to perform the necessary computations so that mAP "
                "and PR curves can be generated"
            ),
        )

        compute_mAP = ctx.params.get("compute_mAP", False)

        if compute_mAP:
            inputs.str(
                "iou_threshs",
                label="IoU thresholds",
                description=(
                    "A comma-separated list of IoU thresholds to use when "
                    "computing mAP"
                ),
            )

            inputs.int(
                "max_preds",
                label="Maximum predictions",
                description=(
                    "A maximum number of predicted objects to evaluate when "
                    "computing mAP and PR curves"
                ),
            )

        _get_instance_mask_parameters(ctx, inputs)

    def parse_parameters(self, ctx, params):
        super().parse_parameters(ctx, params)

        iou_threshs = params.pop("iou_threshs", None)
        if iou_threshs is not None:
            params["iou_threshs"] = [float(t) for t in iou_threshs.split(",")]


class OpenImagesDetection(Detection):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Open Images-style evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#open-images-style-evaluation",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        _get_iscrowd(ctx, inputs)

        inputs.int(
            "max_preds",
            label="Maximum predictions",
            description=(
                "A maximum number of predicted objects to evaluate when "
                "computing mAP and PR curves"
            ),
        )

        _get_instance_mask_parameters(ctx, inputs)

        target_view = get_target_view(ctx, inputs)
        label_fields = _get_label_fields(target_view, fo.Classifications)

        if not label_fields:
            return

        label_field_choices = types.DropdownView()
        for field_name in sorted(label_fields):
            label_field_choices.add_choice(field_name, label=field_name)

        inputs.enum(
            "pos_label_field",
            label_field_choices.values(),
            label="Positive label field",
            description=(
                "A field containing image-level classifications that specify "
                "which classes should be evaluated in the image"
            ),
            view=label_field_choices,
        )

        inputs.enum(
            "neg_label_field",
            label_field_choices.values(),
            label="Negative label field",
            description=(
                "A field containing image-level classifications that specify "
                "which classes should not be evaluated in the image"
            ),
            view=label_field_choices,
        )

    def parse_parameters(self, ctx, params):
        super().parse_parameters(ctx, params)

        iou_threshs = params.pop("iou_threshs", None)
        if iou_threshs is not None:
            params["iou_threshs"] = [float(t) for t in iou_threshs.split(",")]


class ActivityNetDetection(Detection):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="ActivityNet-style evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#activitynet-style-evaluation-default-temporal",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        inputs.bool(
            "compute_mAP",
            default=False,
            label="Compute mAP",
            description=(
                "Whether to perform the necessary computations so that mAP "
                "and PR curves can be generated"
            ),
        )

        compute_mAP = ctx.params.get("compute_mAP", False)

        if compute_mAP:
            inputs.str(
                "iou_threshs",
                label="IoU thresholds",
                description=(
                    "A comma-separated list of IoU thresholds to use when "
                    "computing mAP"
                ),
            )

    def parse_parameters(self, ctx, params):
        super().parse_parameters(ctx, params)

        iou_threshs = params.pop("iou_threshs", None)
        if iou_threshs is not None:
            params["iou_threshs"] = [float(t) for t in iou_threshs.split(",")]


class PlotlyPlotOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="plotly_plot_operator",
            label="Plotly Plot Operator",
            description="Allows users to create various types of Plotly plots",
            dynamic=True
        )
    
    def get_number_field_choices(self, ctx):
        fields = types.Choices(space=6)
        schemas = ctx.dataset.get_field_schema([fof.FloatField, fof.IntField], flat=True)
        for field_path in schemas.keys():
            fields.add_choice(field_path, label=field_path)
        return fields


    def create_axis_input(self, ctx, inputs, axis, is_heatmap):
        axis_obj = types.Object()
        axis_bool_view = types.CheckboxView(space=3)
        axis_obj.str('title', default=None, label=f"Title")
        axis_obj.bool('showgrid', default=True, label="Show Grid", view=axis_bool_view)
        axis_obj.bool('zeroline', default=False, label="Show Zero Line", view=axis_bool_view)
        axis_obj.bool('showline', default=True, label="Show Line", view=axis_bool_view)
        axis_obj.bool('mirror', default=False, label="Mirror", view=axis_bool_view)
        axis_obj.bool('autotick', default=True, label="Auto Tick", view=axis_bool_view)
        axis_obj.bool('showticklabels', default=True, label="Show Tick Labels", view=axis_bool_view)
        axis_obj.bool('showspikes', default=False, label="Show Spikes", view=axis_bool_view)

        if not is_heatmap:
            scale_choices = types.Choices()
            scale_choices.add_choice("linear", label="Linear")
            scale_choices.add_choice("log", label="Log")
            scale_choices.add_choice("date", label="Date")
            scale_choices.add_choice("category", label="Category")
            axis_obj.enum('type', values=scale_choices.values(), view=scale_choices, default="linear", label="Scale Type")

        axis_obj.float('tickangle', default=0, label="Tick Angle")
        axis_obj.str('tickformat', default=None, label="Tick Format", view=types.View(space=3))

        # Range settings
        axis_obj.bool('autorange', default=True, label="Auto Range", view=axis_bool_view)
        autorange = ctx.params.get(f"{axis}axis", {}).get('autorange', True)
        # if not autorange:
            # axis_obj.array('range', element_type=types.Number(), default=None, label="Range", view=types.View(space=3))

        # todo - fix, this should not be a bool
        # axis_obj.bool('showexponent', default="all", label="Show Exponent")
        inputs.define_property(f"{axis}axis", axis_obj, label=f"{axis.capitalize()} Axis")

    def resolve_input(self, ctx):
        prompt = types.PromptView(submit_button_label="Create Plot")
        inputs = types.Object()
        inputs.str('plot_title', label='Plot Title', view=types.View(space=6))
        plot_choices = types.Choices(label="Plot Type", space=6)
        plot_choices.add_choice("histogram", label="Histogram")
        plot_choices.add_choice("line", label="Line")
        plot_choices.add_choice("scatter", label="Scatter")
        plot_choices.add_choice("eval_results", label="Evaluation Results")
        plot_choices.add_choice("heatmap", label="Heatmap")
        plot_choices.add_choice("bar", label="Bar")

        plot_type = ctx.params.get('plot_type')

        inputs.enum('plot_type', values=plot_choices.values(), view=plot_choices, required=True)
        is_heatmap = plot_type == 'heatmap' or plot_type == 'eval_results'

        if not plot_type:
            return types.Property(inputs, view=prompt)

        self.create_axis_input(ctx, inputs, "x", is_heatmap)
        self.create_axis_input(ctx, inputs, "y", is_heatmap)

        fields = self.get_number_field_choices(ctx)
        if not is_heatmap:
            inputs.enum('x_field', values=fields.values(), view=fields, required=True, label="X Data Source")

        if plot_type == 'eval_results':
            eval_keys = types.Choices()
            for key in ctx.dataset.list_evaluations():
                eval_keys.add_choice(key, label=key)
            inputs.enum('eval_key', values=eval_keys.values(), view=eval_keys, required=True, label="Evaluation Key")

        if plot_type == 'histogram':
            inputs.int('bins', default=10, label="Number of Bins", view=types.View(space=6))
            inputs.obj('color', label="Color", default={"hex": '#ff0000'}, view=types.ColorView(compact=True))
            inputs.bool('show_legend', default=True, label="Show Legend")
            inputs.str('legend_title', default='Data', label="Legend Title")

            

            binning_function_choices = types.Choices()
            binning_function_choices.add_choice("count", label="Count")
            binning_function_choices.add_choice("sum", label="Sum")
            binning_function_choices.add_choice("avg", label="Average")
            binning_function_choices.add_choice("min", label="Minimum")
            binning_function_choices.add_choice("max", label="Maximum")
            inputs.enum('binning_function', values=binning_function_choices.values(), view=binning_function_choices, default="count", label="Binning Function")

            normalization_choices = types.Choices()
            normalization_choices.add_choice("", label="None")
            normalization_choices.add_choice("percent", label="Percent")
            normalization_choices.add_choice("probability", label="Probability")
            normalization_choices.add_choice("density", label="Density")
            normalization_choices.add_choice("probability density", label="Probability Density")
            inputs.enum('normalization', values=normalization_choices.values(), view=normalization_choices, default="", label="Normalization")

            inputs.bool('cumulative', default=False, label="Cumulative")

            histfunc_choices = types.Choices()
            histfunc_choices.add_choice("count", label="Count")
            histfunc_choices.add_choice("sum", label="Sum")
            histfunc_choices.add_choice("avg", label="Average")
            histfunc_choices.add_choice("min", label="Minimum")
            histfunc_choices.add_choice("max", label="Maximum")
            inputs.enum('histfunc', values=histfunc_choices.values(), view=histfunc_choices, default="count", label="Histogram Function")

        if plot_type == 'line' or plot_type == 'scatter':
            inputs.enum('y_field', values=fields.values(), view=fields, required=True, label="Y Data Source")

        return types.Property(inputs, view=prompt)
    
    def execute(self, ctx):
        plot_config = ctx.params
        plot_config.pop('panel_state') # todo: remove this and panel_state from params
        return {"plot_config": plot_config}

def _get_iscrowd(ctx, inputs, default=None):
    target = ctx.params.get("target", None)
    target_view = _get_target_view(ctx, target)

    gt_field = ctx.params["gt_field"]
    root, _ = target_view._get_label_field_root(gt_field)

    field = target_view.get_field(root, leaf=True)
    schema = field.get_field_schema(
        ftype=(fo.BooleanField, fo.IntField, fo.FloatField)
    )

    crowd_attrs = sorted(
        k for k, v in schema.items() if k not in ("confidence", "index")
    )

    if crowd_attrs:
        default = crowd_attrs[0]

        crowd_attr_choices = types.AutocompleteView()
        for crowd_attr in crowd_attrs:
            crowd_attr_choices.add_choice(crowd_attr, label=crowd_attr)
    else:
        crowd_attr_choices = None

    inputs.str(
        "iscrowd",
        default=default,
        label="Crowd attribute",
        description="The name of the crowd attribute (if any)",
        view=crowd_attr_choices,
    )


def _get_instance_mask_parameters(ctx, inputs):
    inputs.int(
        "tolerance",
        label="Tolerance",
        description=(
            "A tolerance, in pixels, when generating approximate "
            "polylines for instance masks. Typical values are 1-3 pixels"
        ),
    )

    error_level_choices = types.DropdownView()
    error_level_choices.add_choice(
        0,
        label="0: Raise geometric errors that are encountered",
    )
    error_level_choices.add_choice(
        1,
        label="1: Log warnings if geometric errors are encountered",
    )
    error_level_choices.add_choice(
        2,
        label="2: Ignore geometric errors",
    )

    inputs.enum(
        "error_level",
        error_level_choices.values(),
        default=1,
        label="Error level",
        description=(
            "The error level to use when manipulating instance masks or "
            "polylines. If error level is > 0, any calculation that "
            "raises a geometric error will default to an IoU of 0"
        ),
        view=error_level_choices,
    )


class Segmentation(EvaluationMethod):
    def get_parameters(self, ctx, inputs):
        _get_mask_targets(ctx, inputs)

    def parse_parameters(self, ctx, params):
        mask_targets = params.pop("mask_targets", None)
        if mask_targets is not None:
            params["mask_targets"] = _parse_mask_targets(ctx, mask_targets)


class SimpleSegmentation(Segmentation):
    def get_parameters(self, ctx, inputs):
        inputs.view(
            "parameters",
            types.Header(
                label="Simple segmentation evaluation",
                description="https://docs.voxel51.com/user_guide/evaluation.html#id17",
                divider=True,
            ),
        )

        super().get_parameters(ctx, inputs)

        inputs.int(
            "bandwidth",
            label="Bandwidth",
            description=(
                "An optional bandwidth along the contours of the ground truth "
                "masks to which to restrict attention when computing "
                "accuracies. A typical value for this parameter is 5 pixels. "
                "By default, the entire masks are evaluated"
            ),
        )

        average_choices = types.DropdownView()
        average_choices.add_choice(
            "micro",
            label="micro",
            description=(
                "Calculate metrics globally by considering each element of "
                "the label indicator matrix as a label"
            ),
        )
        average_choices.add_choice(
            "macro",
            label="macro",
            description=(
                "Calculate metrics for each label, and find their unweighted "
                "mean. This does not take label imbalance into account"
            ),
        )
        average_choices.add_choice(
            "weighted",
            label="weighted",
            description=(
                "Calculate metrics for each label, and find their average, "
                "weighted by support (the number of true instances for each "
                "label"
            ),
        )
        average_choices.add_choice(
            "samples",
            label="samples",
            description=(
                "Calculate metrics for each instance, and find their average"
            ),
        )

        inputs.enum(
            "average",
            average_choices.values(),
            required=True,
            label="Average",
            description=(
                "The averaging strategy to use when populating precision and "
                "recall numbers on each sample"
            ),
            view=average_choices,
        )


def _get_classes(ctx, inputs):
    dataset = ctx.dataset
    gt_field = ctx.params["gt_field"]
    pred_field = ctx.params["pred_field"]

    classes_choices = types.DropdownView()
    if dataset.default_classes:
        classes_choices.add_choice("default", label="default classes")

    if gt_field in dataset.classes:
        classes_choices.add_choice("gt", label=f"{gt_field} classes")

    if pred_field in dataset.classes:
        classes_choices.add_choice("pred", label=f"{pred_field} classes")

    if classes_choices:
        inputs.enum(
            "classes",
            classes_choices.values(),
            label="Classes",
            description=(
                "If you have class list(s) stored on your dataset, you can "
                "select one to use below. If not provided, the observed "
                "ground truth/predicted labels are used"
            ),
            view=classes_choices,
        )


def _parse_classes(ctx, classes):
    if classes == "default":
        return ctx.dataset.default_classes

    if classes == "gt":
        gt_field = ctx.params["gt_field"]
        return ctx.dataset.classes[gt_field]

    if classes == "pred":
        pred_field = ctx.params["pred_field"]
        return ctx.dataset.classes[pred_field]

    return None


def _get_mask_targets(ctx, inputs):
    dataset = ctx.dataset
    gt_field = ctx.params["gt_field"]
    pred_field = ctx.params["pred_field"]

    mask_targets_choices = types.DropdownView()
    if dataset.default_classes:
        mask_targets_choices.add_choice("default", label="default classes")

    if gt_field in dataset.classes:
        mask_targets_choices.add_choice("gt", label=f"{gt_field} classes")

    if pred_field in dataset.classes:
        mask_targets_choices.add_choice("pred", label=f"{pred_field} classes")

    if mask_targets_choices:
        inputs.enum(
            "mask_targets",
            mask_targets_choices.values(),
            label="Mask targets",
            description=(
                "If you have mask target(s) stored on your dataset, you can "
                "select one to use below. If not provided, the observed "
                "ground truth/predicted labels are used"
            ),
            view=mask_targets_choices,
        )


def _parse_mask_targets(ctx, mask_targets):
    if mask_targets == "default":
        return ctx.dataset.default_mask_targets

    if mask_targets == "gt":
        gt_field = ctx.params["gt_field"]
        return ctx.dataset.mask_targets[gt_field]

    if mask_targets == "pred":
        pred_field = ctx.params["pred_field"]
        return ctx.dataset.mask_targets[pred_field]

    return None


def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def get_new_eval_key(
    ctx,
    inputs,
    name="eval_key",
    label="Evaluation key",
    description="Provide an evaluation key for this run",
):
    prop = inputs.str(
        name,
        required=True,
        label=label,
        description=description,
    )

    eval_key = ctx.params.get(name, None)
    if eval_key is not None and eval_key in ctx.dataset.list_evaluations():
        prop.invalid = True
        prop.error_message = "Evaluation key already exists"
        eval_key = None

    return eval_key

def get_axis_config(plot_config, axis):
    axis_config = plot_config.get(f"{axis}axis", None)    
    return axis_config

def get_plot_view(plot_config, on_click=None):
    panel_id = plot_config.get('panel_id') # this should come from ctx (not params)
    plot_type = plot_config.get('plot_type')
    plot_title = plot_config.get('plot_title', None)
    config = {}
    show_legend = plot_config.get('show_legend', False)
    layout = {
        "title": plot_title,
        "xaxis": get_axis_config(plot_config, "x"),
        "yaxis": get_axis_config(plot_config, "y"),
        "showlegend": show_legend,
    }
    x_field = plot_config.get('x_field')
    y_field = plot_config.get('y_field')
    # is_line = plot_type == "line"
    # is_scatter = plot_type == "scatter"
    # is_eval_results = plot_type == "eval_results"

    actual_title = plot_title
    if not plot_title:
        if x_field:
            actual_title = f"{plot_title} ({x_field})"
        if y_field:
            actual_title = f"{plot_title} ({x_field}, {y_field})"

    # Construct the Plotly view (note: panel_id should be auto injected somehow)
    plotly_view = types.PlotlyView(
        panel_id=panel_id,
        label=actual_title,
        config=config,
        layout=layout,
        on_click=on_click,
        x_data_source=plot_config.get('x_field'),
        show_selected=True,
        height=85
        # add a param to allow for pulling in ids in callbacks
        # include_point_ids_in_callback=True
    )

    return plotly_view

def set_view_for_confustion_matrix_cell(ctx, x_field, y_field, x, y):
    view = ctx.dataset.filter_labels(x_field, F("label") == x)
    view = view.filter_labels(y_field, F("label") == y)
    ctx.ops.set_view(view)

def get_plot_data(sample_collection, plot_config):
    data = []
    plot_type = plot_config.get('plot_type')
    x_field = plot_config.get('x_field')
    is_line = plot_type == "line"
    is_scatter = plot_type == "scatter"
    if plot_type == "histogram":
        bins = plot_config.get('bins', 10)
        binning_function = plot_config.get('binning_function', 'count')
        normalization = plot_config.get('normalization', None)
        cumulative = plot_config.get('cumulative', False)
        histfunc = plot_config.get('histfunc', 'count')
        data = [{
            "type": "histogram",
            "x": sample_collection.values(x_field),
            "nbinsx": bins,
            "histfunc": histfunc,
            "cumulative_enabled": cumulative,
            # "marker": {"color": color},
            "xbin": {"func": binning_function},
            "histnorm": normalization,
        }]
    elif is_line or is_scatter:
        data = [{
            "type": "scatter",
            "mode": is_line and "lines" or "markers",
            "x": sample_collection.values(plot_config.get('x_field')),
            "y": sample_collection.values(plot_config.get('y_field'))
        }]
    elif plot_type == "heatmap":
        data =   [{
            "z": [[1, null, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
            "x": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            "y": ['Morning', 'Afternoon', 'Evening'],
            "type": 'heatmap',
            "hoverongaps": False
        }]
    elif plot_type == "bar":
        data = [{"type": "bar", "x": [1, 2, 3], "y": [1, 4, 9]}]
    elif plot_type == "eval_results":
        eval_key = plot_config.get('eval_key')
        eval_results = sample_collection.load_evaluation_results(eval_key)
        confusion_matrix = eval_results.confusion_matrix().tolist()
        labels = list(eval_results.classes)

        print('labels length', len(labels))
        print('confusion matrix length', len(confusion_matrix))

        data = [{
            "type": "heatmap",
            "z": confusion_matrix,
            "x": labels,
            "y": labels
        }]

    return data

class AutoLaunchPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="try_fiftyone_auto_launch_panel",
            label="Try FiftyOne: Auto Launch Panel",
            unlisted=True,
            on_dataset_open=True,
        )

    def execute(self, ctx):
        auto_launch_panel = ctx.dataset.app_config.plugins.get(
            self.plugin_name, {}
        ).get("auto_launch_panel", None)
        if auto_launch_panel:
            ctx.ops.open_panel(auto_launch_panel, layout="horizontal", force=True)
        return {}

def add_panel_navigation(panel, left=True, right=False, on_left=None, on_right=None):
    base_btn_styles = {
        "position": "absolute",
        "top": "50%",
        "minWidth": 0,
        "padding": "8px",
        "background":"#333333",
        "&:hover": {
            "background": "#2b2a2a"
        }
    }
    if left:
        panel.btn(
            "previous",
            label="Previous",
            icon="arrow_back",
            variant="contained",
            componentsProps={ "button": { "sx": { **base_btn_styles, "left": 8 }}},
            on_click=on_left,
        )
    if right:
        panel.btn(
            "next",
            label="Next",
            icon="arrow_forward",
            variant="contained",
            componentsProps={ "button": { "sx": { **base_btn_styles, "right": 8 }}},
            on_click=on_right,
        )

def register(p):
    p.register(EvaluationPanel)


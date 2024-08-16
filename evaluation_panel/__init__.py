import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.brain as fob
from fiftyone.brain import Similarity
import fiftyone as fo
import fiftyone.zoo.models as fozm
import fiftyone.core.patches as fop
from fiftyone import ViewField as F
import fiftyone.core.fields as fof
import numpy as np
import fiftyone.core.labels as fol



class EvaluationPanel(foo.Panel):

    @property
    def config(self):
        return foo.PanelConfig(
            name="evaluation_panel",
            label="Evalution Panel",
            icon="assessment"
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
        
        eval_comp = stack.menu("eval_comp", variant="contained")
        # Add operator buttons
        eval_comp.btn("evaluate_model", label="Evaluate Model", on_click=self.evaluate_model)
        eval_comp.btn("compare_models", label="Compare Models", on_click=self.compare_models)


        # Create the eval key options for the menus
        keys = []
        for key in ctx.dataset.list_evaluations():
            keys.append(key)
        current_eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        current_compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key")
        eval_keys = keys.copy()
        compare_keys = keys.copy()
        if current_compare_key in eval_keys:
            eval_keys.remove(current_compare_key)
        if current_eval_key in compare_keys:
            compare_keys.remove(current_eval_key)
        menu = stack.menu('menu', variant="square", width=100, align_y="center")
        actions = menu.btn_group('actions')

        # Add Eval Key Menu
        actions.enum('eval_key', label="Evaluation key", values=eval_keys,  view=types.View(space=3), on_change=self.on_change_config,)

        

        # # Check to see if state initialized the stack
        # if ctx.panel.get_state("my_stack") != None:
        #     eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        #     if eval_key != None:

        #         # Remove active eval key from the compare list
        #         compare_keys = eval_keys.values()
        #         index = compare_keys.index(eval_key )
        #         compare_keys.pop(index)

        #         # Find the type of eval
        #         current_eval = ctx.dataset.get_evaluation_info(eval_key)
        #         eval_type = current_eval.config.type

            
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
        ctx.panel.set_data("my_stack.evaluations", table_list) 

        if eval_type == "detection":
            if current_eval["config"]["compute_mAP"]:
                mAP_list = []
                for i,label in enumerate(results.classes):
                    new_row = {"class": label, "AP": results._classwise_AP[i]}
                    mAP_list.append(new_row)

                new_row = {"class": "All", "AP": results.mAP()}
                mAP_list.append(new_row)
                ctx.panel.set_data("my_stack.mAP_evaluations", mAP_list) 

class EvalPlots(foo.Panel):

    @property
    def config(self):
        return foo.PanelConfig(
            name="eval_plots",
            label="eval_plots"
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
            ### Evaluate Plots
        """,
        name="md1")

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

                self._add_plots(ctx,stack)

            
        # Add Compare Key menu
        actions.enum('compare_key', label="Compare key", values=compare_keys,  view=types.View(space=3), on_change=self.on_change_config,)

        

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

    def on_change_config(self, ctx):
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        current_eval = ctx.dataset.get_evaluation_info(eval_key).serialize()
        eval_type = current_eval["config"]["type"]
        results = ctx.dataset.load_evaluation_results(eval_key)
        report = results.report()
        table_list = []
        self._update_plot_data(ctx)

    def _update_plot_data(self, ctx,):
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        results = ctx.dataset.load_evaluation_results(eval_key)
        pred_field = info["config"]["pred_field"]
        gt_field = info["config"]["gt_field"]
        

        if info["config"]["type"] != "classification":

            classes = ctx.dataset.distinct(f"{pred_field}.detections.label")

            x = f"{pred_field}.detections.confidence"
            bins = 10
            counts, edges, other = ctx.view.histogram_values(
                x,
                bins=bins,
            )
            counts = np.asarray(counts)
            edges = np.asarray(edges)
            
            left_edges = edges[:-1]
            widths = edges[1:] - edges[:-1]
            histogram_data = {
                "name": f"{pred_field}.detections.confidence",
                'x': left_edges.tolist(),
                'y': counts.tolist(),
                'ids': ["01","02","03","04","05","06","07","08","09", "10"],
                'type': 'bar',
                'width': widths.tolist()
            }
            ctx.panel.set_data('my_stack.confidence', [
                # trace
                histogram_data,
            ])

            x = f"{pred_field}.detections.{eval_key}_iou"
            bins = 10
            counts, edges, other = ctx.view.histogram_values(
                x,
                bins=bins,
            )
            counts = np.asarray(counts)
            edges = np.asarray(edges)
            
            left_edges = edges[:-1]
            widths = edges[1:] - edges[:-1]
            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}_iou",
                'x': left_edges.tolist(),
                'y': counts.tolist(),
                'type': 'bar',
                'width': widths.tolist()
            }
            ctx.panel.set_data('my_stack.iou', [
                # trace
                histogram_data,
            ])

            tp = np.array(ctx.dataset.values(f"{eval_key}_tp"))
            fp = np.array(ctx.dataset.values(f"{eval_key}_fp"))
            fn = np.array(ctx.dataset.values(f"{eval_key}_fn"))

            p = tp / (tp + fp)
            p = np.nan_to_num(p, nan=0.0)
            r = tp / (tp + fn)
            r = np.nan_to_num(r, nan=0.0)
            f1 = 2 * (p * r) / (p + r)
            f1 = np.nan_to_num(f1, nan=0.0)

            p_left_edges, p_counts, p_widths = compute_histogram(p,10)

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': p_left_edges.tolist(),
                'y': p_counts.tolist(),
                'type': 'bar',
                'width': p_widths.tolist()
            }
            ctx.panel.set_data('my_stack.precision', [
                # trace
                histogram_data,
            ])

            r_left_edges, r_counts, r_widths = compute_histogram(r,10)

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': r_left_edges.tolist(),
                'y': r_counts.tolist(),
                'type': 'bar',
                'width': r_widths.tolist()
            }
            ctx.panel.set_data('my_stack.recall', [
                # trace
                histogram_data,
            ])

            f1_left_edges, f1_counts, f1_widths = compute_histogram(f1,10)

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': f1_left_edges.tolist(),
                'y': f1_counts.tolist(),
                'type': 'bar',
                'width': f1_widths.tolist()
            }
            ctx.panel.set_data('my_stack.f1', [
                # trace
                histogram_data,
           ])
            
            p_class_list = []
            r_class_list = []
            f1_class_list = []
            for cls in classes:

                tp = sum(sublist.count("tp") for sublist in ctx.dataset.filter_labels(pred_field, F("label").is_in([cls])).values(f"{pred_field}.detections.{eval_key}"))
                fp = sum(sublist.count("fp") for sublist in ctx.dataset.filter_labels(pred_field, F("label").is_in([cls])).values(f"{pred_field}.detections.{eval_key}"))
                fn = sum(sublist.count("fn") for sublist in ctx.dataset.filter_labels(gt_field, F("label").is_in([cls])).values(f"{gt_field}.detections.{eval_key}"))


                p = tp / (tp + fp)
                p = np.nan_to_num(p, nan=0.0)
                r = tp / (tp + fn)
                r = np.nan_to_num(r, nan=0.0)
                f1 = 2 * (p * r) / (p + r)
                f1 = np.nan_to_num(f1, nan=0.0)
            
                p_class_list.append(p)
                r_class_list.append(r)
                f1_class_list.append(f1)
            
            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': p_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.p_class', [
                # trace
                histogram_data,
            ])

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': r_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.r_class', [
                # trace
                histogram_data,
            ])

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': f1_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.f1_class', [
                # trace
                histogram_data,
            ])
        elif  info["config"]["type"] == "classification":

            classes = ctx.dataset.distinct(f"{pred_field}.label")

            x = f"{pred_field}.confidence"
            bins = 10
            counts, edges, other = ctx.view.histogram_values(
                x,
                bins=bins,
            )
            counts = np.asarray(counts)
            edges = np.asarray(edges)
            
            left_edges = edges[:-1]
            widths = edges[1:] - edges[:-1]
            histogram_data = {
                "name": f"{pred_field}.confidence",
                'x': left_edges.tolist(),
                'ids': ["01","02","03","04","05","06","07","08","09", "10"],
                'y': counts.tolist(),
                'type': 'bar',
                'width': widths.tolist()
            }
            ctx.panel.set_data('my_stack.confidence', [
                # trace
                histogram_data,
            ])

            p_class_list = []
            r_class_list = []
            f1_class_list = []
            conf_class_list = []
            for cls in classes:

                tp = ctx.dataset.filter_labels(pred_field, F("label").is_in([cls])).values(f"{eval_key}").count(True)
                fp = ctx.dataset.filter_labels(pred_field, F("label").is_in([cls])).values(f"{eval_key}").count(False)
                fn = ctx.dataset.filter_labels(gt_field, F("label").is_in([cls])).values(f"{eval_key}").count(False)



                p = tp / (tp + fp)
                p = np.nan_to_num(p, nan=0.0)
                r = tp / (tp + fn)
                r = np.nan_to_num(r, nan=0.0)
                f1 = 2 * (p * r) / (p + r)
                f1 = np.nan_to_num(f1, nan=0.0)
            
                p_class_list.append(p)
                r_class_list.append(r)
                f1_class_list.append(f1)
            
            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': p_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.p_class', [
                # trace
                histogram_data,
            ])

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': r_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.r_class', [
                # trace
                histogram_data,
            ])

            histogram_data = {
                "name": f"{pred_field}.detections.{eval_key}",
                'x': classes,
                'y': f1_class_list,
                'type': 'bar',
            }
            ctx.panel.set_data('my_stack.f1_class', [
                # trace
                histogram_data,
            ])


        confusion_matrix, labels, ids = results._confusion_matrix(include_other=False, include_missing=False,)
        confusion_matrix = np.asarray(confusion_matrix)
        ids = np.asarray(ids)
        num_rows, num_cols = confusion_matrix.shape
        zlim = [0, confusion_matrix.max()]
        eval_info = ctx.dataset.get_evaluation_info(eval_key)
        gt_field = eval_info.config.gt_field
        pred_field = eval_info.config.pred_field
        label_type = ctx.dataset._get_label_field_type(gt_field)
        use_patches = issubclass(label_type, (fol.Detections, fol.Polylines))
        label_fields = [gt_field, pred_field]
        xlabels = labels[:num_cols]
        ylabels = labels[:num_rows]

        # Flip data so plot will have the standard descending diagonal
        # Flipping the yaxis via `autorange="reversed"` isn't an option because
        # screenshots don't seem to respect that setting...


        if use_patches:
            selection_mode = "patches"
            init_fcn = lambda view: view.to_evaluation_patches(eval_key)
        else:
            selection_mode = "select"
            init_fcn = None

        cm_data = {
            'z': confusion_matrix,
            'x': labels,
            'y': labels,
            'type': 'heatmap',
        }
        ctx.panel.set_data('my_stack.cm', [
            # trace
            cm_data,
        ])


        

    def on_numerical_click(self,ctx):
        print(ctx.params)
        path = ctx.params.get("path")
        plot = ctx.panel.get_state(path)
        raise Exception(plot)
        """
        params = ctx.panel.state.get("my_stack.data")
        x = params["x"]
        left_edge
        right_edge
        plot_clicked

        bar_click
        bars_clicked

        name = params["name"]
        view = ctx.dataset.match(name < x)
        ctx.ops.set_view(view)
        """


    def on_categorical_click(self,ctx):
        pass
        
    def _add_plots(self, ctx, stack):
    
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        
            
        current_eval = ctx.dataset.get_evaluation_info(eval_key).serialize()

        
        config = {
                "scrollZoom": False
            }
        layout = {
            "title": "Confidence",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {
                "title": "Confidence"
            },
            "yaxis": {
                "title": "Count"
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1,
                "showlegend": True
            }
        }
        stack.add_property('confidence', types.Property(
            types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_numerical_click, on_selected=self.on_numerical_click)
        ))


        if current_eval["config"]["type"] != "classification":
            config = {
                "scrollZoom": False
            }
            layout = {
                "title": "IOU",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {
                    "title": "IOU"
                },
                "yaxis": {
                    "title": "Count"
                },
                "showlegend": True,
                "legend": {
                    "x": 0,
                    "y": 1,
                    "showlegend": True
                }
            }
            stack.add_property('iou', types.Property(
                types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_numerical_click)
            ))

            config = {
                "scrollZoom": False
            }
            layout = {
                "title": "Precision Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {
                    "title": "Precision per Sample"
                },
                "yaxis": {
                    "title": "Count"
                },
                "showlegend": True,
                "legend": {
                    "x": 0,
                    "y": 1,
                    "showlegend": True
                }
            }
            stack.add_property('precision', types.Property(
                types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_numerical_click)
            ))

            
            config = {
                "scrollZoom": False
            }
            layout = {
                "title": "Recall Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {
                    "title": "Recall per Sample"
                },
                "yaxis": {
                    "title": "Count"
                },
                "showlegend": True,
                "legend": {
                    "x": 0,
                    "y": 1,
                    "showlegend": True
                }
            }
            stack.add_property('recall', types.Property(
                types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_numerical_click)
            ))

            config = {
                "scrollZoom": False
            }
            layout = {
                "title": "F1-Score Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {
                    "title": "F1-Score per Sample"
                },
                "yaxis": {
                    "title": "Count"
                },
                "showlegend": True,
                "legend": {
                    "x": 0,
                    "y": 1,
                    "showlegend": True
                }
            }
            stack.add_property('f1', types.Property(
                types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_numerical_click)
            ))

        config = {
            "scrollZoom": False
        }
        layout = {
            "title": "Precision per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {
                "title": "Precision per Class"
            },
            "yaxis": {
                "title": "Precision"
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1,
                "showlegend": True
            }
        }
        stack.add_property('p_class', types.Property(
            types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_categorical_click)
        ))

        config = {
            "scrollZoom": False
        }
        layout = {
            "title": "Recall per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {
                "title": "Recall per Class"
            },
            "yaxis": {
                "title": "Recall"
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1,
                "showlegend": True
            }
        }
        stack.add_property('r_class', types.Property(
            types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_categorical_click)
        ))

        config = {
            "scrollZoom": False
        }
        layout = {
            "title": "F1-Score per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {
                "title": "F1-Score per Class"
            },
            "yaxis": {
                "title": "F1-Score"
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1,
                "showlegend": True
            }
        }
        stack.add_property('f1_class', types.Property(
            types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout, on_click=self.on_categorical_click)
        ))

        config = {}
        layout = {
                    "title": "Confusion Matrix",
                    "xaxis": {"fixedrange": True, "title": "Ground truth"},
                    "yaxis": {"fixedrange": True, "title": "Model predictions"},
                }
        
        stack.add_property('cm', types.Property(
            types.List(types.Object()), view=types.PlotlyView(config=config, layout=layout,)
        ))

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

def compute_histogram(values, num_bins):
    # Compute the histogram
    counts, bin_edges = np.histogram(values, bins=num_bins)
    
    # Calculate the left edges of the bins
    left_edges = bin_edges[:-1]
    
    # Calculate the width of each bin
    bin_widths = np.diff(bin_edges)
    
    return left_edges, counts, bin_widths



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
    p.register(EvalPlots)


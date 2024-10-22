import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np
from fiftyone import ViewField as F


class EvaluationPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="evaluation_panel", label="Evalution Panel", icon="assessment"
        )

    def on_load(self, ctx):
        # To start, on load we populate the first menu with our current datasets evaluation keys
        keys = []
        for key in ctx.dataset.list_evaluations():
            keys.append(key)
        ctx.panel.set_state("eval_keys", keys)

    def render(self, ctx):
        panel = types.Object()

        # Define main stack
        stack = panel.v_stack("my_stack", align_x="center", gap=2)

        stack.md(
            """
            ### Evaluate Your Models
        """,
            name="md1",
        )

        # Add operator buttons
        eval_comp = stack.menu("eval_comp", variant="contained")
        eval_comp.btn("apply_model", label="Apply Zoo Model", on_click=self.apply_model)
        eval_comp.btn(
            "evaluate_model", label="Evaluate Model", on_click=self.evaluate_model
        )

        # Create the eval key options for the menus
        keys = ctx.panel.get_state("eval_keys")
        current_eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        current_compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key")
        eval_keys = keys.copy()
        compare_keys = keys.copy()
        if current_compare_key in eval_keys:
            eval_keys.remove(current_compare_key)
        if current_eval_key in compare_keys:
            compare_keys.remove(current_eval_key)
        menu = stack.menu("menu", variant="square", width=100, align_y="center")
        actions = menu.btn_group("actions")

        # Add Eval Key Menu
        actions.enum(
            "eval_key",
            label="Evaluation key",
            values=eval_keys,
            view=types.View(space=3),
            on_change=self.on_change_config,
        )
        # Add Compare Key menu
        actions.enum(
            "compare_key",
            label="Compare key",
            values=compare_keys,
            view=types.View(space=3),
            on_change=self.on_change_config,
        )

        # Define Tab View for Eval Results vs Info
        tabs = types.TabsView()
        tabs.add_choice("results", label="Evaluation Results")
        tabs.add_choice("info", label="Evaluation Info")

        stack.str("tabs", view=tabs, on_change=self.on_path_change)

        # Define the paths for tabs
        if ctx.panel.get_state("my_stack") is not None:
            eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
            if eval_key is not None:
                current_tab = ctx.panel.get_state("my_stack.tabs")

                info = ctx.dataset.get_evaluation_info(eval_key)
                if current_tab == "results":
                    _eval_results(
                        ctx,
                        stack,
                    )
                    self._add_plots(ctx, stack)

                elif current_tab == "info":
                    if info:
                        _eval_info(ctx, stack, info)

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

    def evaluate_model(self, ctx):
        # Called when you click the "Evaluate Model" button
        ctx.prompt("@voxel51/evaluation/evaluate_model")
        ctx.panel.state.view = "eval"

    def apply_model(self, ctx):
        # Called when you click the "Apply Zoo Model" button
        ctx.prompt("@voxel51/zoo/apply_zoo_model")
        ctx.panel.state.view = "apply"

    def on_path_change(self, ctx):
        # ctx.trigger("reload_samples")

        print("on_change:", ctx.params)

    def on_change_config(self, ctx):
        # Called when either of the values in the eval key or compare key menus change
        self._update_table_data(ctx)
        self._update_plot_data(ctx)

    def on_numerical_click(self, ctx):
        # Called when you click on a numerical histogram

        # The function then return a view based on the bar you have clicked
        # If there are two traces, it filters both label fields and returns combination of the two views

        # Figure out which plot was clicked
        params = ctx.params
        path = ctx.params.get("path")
        plot = ctx.panel.get_state(path)

        # Grab the range of the values clicked
        left_edge = params["range"][0]
        right_edge = params["range"][1]

        # Check to see if it is a detection or classification plot
        if "detections" in plot[0]["name"]:
            # Grab keys and other relevant info
            eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
            compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
            path = ctx.params.get("path")
            info = ctx.dataset.get_evaluation_info(eval_key).serialize()
            pred_field = info["config"]["pred_field"]

            # If only one trace, return just that range of values as a view
            if len(plot) == 1:
                view = ctx.dataset.filter_labels(
                    pred_field, F(plot[0]["name"]) <= right_edge
                ).filter_labels(
                    pred_field, F(plot[0]["name"].split(".")[-1]) >= left_edge
                )
                ctx.ops.set_view(view)

            # If two traces, then filter all the detections so that only ones in the range are shown
            elif len(plot) == 2:
                c_info = ctx.dataset.get_evaluation_info(compare_key).serialize()
                c_pred_field = c_info["config"]["pred_field"]
                view = (
                    ctx.dataset.filter_labels(
                        pred_field, F(plot[0]["name"].split(".")[-1]) <= right_edge
                    )
                    .filter_labels(
                        pred_field, F(plot[0]["name"].split(".")[-1]) >= left_edge
                    )
                    .filter_labels(
                        c_pred_field, F(plot[1]["name"].split(".")[-1]) <= right_edge
                    )
                    .filter_labels(
                        c_pred_field, F(plot[1]["name"].split(".")[-1]) >= left_edge
                    )
                )
                ctx.ops.set_view(view)
            else:
                raise Exception(plot)

        # Else this is a classification plot
        else:
            # If one trace match the values and return the view
            if len(plot) == 1:
                view = ctx.dataset.match(F(plot[0]["name"]) <= right_edge).match(
                    F(plot[0]["name"]) >= left_edge
                )
                ctx.ops.set_view(view)
            # If two, match for both label fields and return
            elif len(plot) == 2:
                view1 = ctx.dataset.match(F(plot[0]["name"]) <= right_edge).match(
                    F(plot[0]["name"]) >= left_edge
                )
                view2 = ctx.dataset.match(F(plot[0]["name"]) <= right_edge).match(
                    F(plot[0]["name"]) >= left_edge
                )
                view = view1.concat(view2)
                ctx.ops.set_view(view)
            else:
                raise Exception(plot)

    def on_categorical_click(self, ctx):
        # Called when clicking on a categorical histogram
        # Returns the class filtered for that histogram

        # Grab starting info needed for the event
        params = ctx.params
        path = ctx.params.get("path")
        plot = ctx.panel.get_state(path)
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
        path = ctx.params.get("path")
        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        pred_field = info["config"]["pred_field"]

        # If only one trace, return just class of Model 1
        if compare_key is None:
            view = ctx.dataset.filter_labels(
                pred_field, F("label").is_in([params["x"]])
            )
            ctx.ops.set_view(view)

        # If two traces, return the samples of the class for both Model 1 and Model 2
        elif compare_key is not None:
            c_info = ctx.dataset.get_evaluation_info(compare_key).serialize()
            c_pred_field = c_info["config"]["pred_field"]

            view1 = ctx.dataset.filter_labels(
                pred_field, F("label").is_in([params["x"]])
            )
            view2 = ctx.dataset.filter_labels(
                c_pred_field, F("label").is_in([params["x"]])
            )
            view = view1.concat(view2)
            ctx.ops.set_view(view)
        else:
            raise Exception(plot)

    def on_cm_click(self, ctx):
        # Called when you click on a confusion matrix
        # Returns the respective values of the Model X and the GT filtered

        # Start by grabbing our needed values
        params = ctx.params
        path = ctx.params.get("path")
        plot = ctx.panel.get_state(path)
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
        path = ctx.params.get("path")
        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        pred_field = info["config"]["pred_field"]
        gt_field = info["config"]["gt_field"]
        eval_type = info["config"]["type"]

        # If this is Model 1's confusion matrix, return based on Model 1's fields
        if "c_cm" not in path:
            # If x == None, we are looking at false negatives in the GT
            if params["x"] == "(none)":
                view = ctx.dataset.filter_labels(
                    gt_field, F("label").is_in([params["y"]])
                ).filter_labels(gt_field, F(eval_key).is_in(["fn"]))
            # If y is none, we are looking at false positives in the pred field
            elif params["y"] == "(none)":
                view = ctx.dataset.filter_labels(
                    pred_field, F("label").is_in([params["x"]])
                ).filter_labels(pred_field, F(eval_key).is_in(["fp"]))
            # If neither are none, we can return the filtered labels for both GT and Model X
            else:
                view = ctx.dataset.filter_labels(
                    gt_field, F("label").is_in([params["y"]])
                ).filter_labels(pred_field, F("label").is_in([params["x"]]))
            ctx.ops.set_view(view)

        # If this is Model 2's confusion matrix, return based on Model 2's fields
        else:
            c_info = ctx.dataset.get_evaluation_info(compare_key).serialize()
            c_pred_field = c_info["config"]["pred_field"]
            c_gt_field = c_info["config"]["gt_field"]

            # If x == None, we are looking at false negatives in the GT
            if params["x"] == "(none)":
                view = ctx.dataset.filter_labels(
                    c_gt_field, F("label").is_in([params["y"]])
                ).filter_labels(c_gt_field, F(eval_key).is_in(["fn"]))
            # If y is none, we are looking at false positives in the pred field
            elif params["y"] == "(none)":
                view = ctx.dataset.filter_labels(
                    c_pred_field, F("label").is_in([params["x"]])
                ).filter_labels(c_pred_field, F(eval_key).is_in(["fp"]))
            # If neither are none, we can return the filtered labels for both GT and Model X
            else:
                view = ctx.dataset.filter_labels(
                    c_gt_field, F("label").is_in([params["y"]])
                ).filter_labels(c_pred_field, F("label").is_in([params["x"]]))
            ctx.ops.set_view(view)

    def _update_table_data(self, ctx):
        # Updates the DATA going into the table, this is called in on_change_config
        # Any time the menus values change, update all values going into the table
        # If compare key does not exist, update the (table_name)
        # If compare key does exist, update the c_(table_name)

        # Start by grabbing the basics
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        eval_type = info["config"]["type"]
        results = ctx.dataset.load_evaluation_results(eval_key)
        report = results.report()

        table_list = []

        # Single eval path, update (table name)
        if compare_key is None:
            if eval_type == "classification":
                for key in report:
                    # TODO
                    # We skip accuracy as it doesnt fit into our table
                    if key != "accuracy":
                        table_list.append(
                            {
                                "class": key,
                                "precision": int(report[key]["precision"] * 1000)
                                / 1000,
                                "recall": int(report[key]["recall"] * 1000) / 1000,
                                "f1-score": int(report[key]["f1-score"] * 1000) / 1000,
                                "support": report[key]["support"],
                            }
                        )
                ctx.panel.set_data("my_stack.evaluations", table_list)

            elif eval_type == "detection":
                for key in report:
                    table_list.append(
                        {
                            "class": key,
                            "precision": int(report[key]["precision"] * 1000) / 1000,
                            "recall": int(report[key]["recall"] * 1000) / 1000,
                            "f1-score": int(report[key]["f1-score"] * 1000) / 1000,
                            "support": report[key]["support"],
                        }
                    )
                ctx.panel.set_data("my_stack.evaluations", table_list)

                # Not every detections eval has computed mAP scores,
                # so check if the eval_info states that it is there and update accordingly
                if info["config"]["compute_mAP"]:
                    mAP_list = []
                    for i, label in enumerate(results.classes):
                        new_row = {
                            "class": label,
                            "AP": int(results._classwise_AP[i] * 1000) / 1000,
                        }
                        mAP_list.append(new_row)

                    new_row = {"class": "All", "AP": int(results.mAP() * 1000) / 1000}
                    mAP_list.append(new_row)
                    ctx.panel.set_data("my_stack.mAP_evaluations", mAP_list)

        # Compare key DOES exist, update c_(table_name) instead
        else:
            c_eval = ctx.dataset.get_evaluation_info(compare_key).serialize()
            c_results = ctx.dataset.load_evaluation_results(compare_key)
            c_report = c_results.report()
            c_keys = list(c_report.keys())

            if eval_type == "classification":
                for key in report:
                    if key != "accuracy":
                        # Not every eval will be the same, if the class doesnt exist in compare_eval
                        # add a 0 to that row instead.

                        # NOTE, we are going off eval_key as the source of truth
                        # If eval_key has 1 class and compare_key has 10, we will not see the other 9
                        # of compare_key, only if you flip the values will you see it
                        if key in c_keys:
                            table_list.append(
                                {
                                    "class": key,
                                    "precision": int(report[key]["precision"] * 1000)
                                    / 1000,
                                    "c_precision": int(
                                        c_report[key]["precision"] * 1000
                                    )
                                    / 1000,
                                    "recall": int(report[key]["recall"] * 1000) / 1000,
                                    "c_recall": int(c_report[key]["recall"] * 1000)
                                    / 1000,
                                    "f1-score": int(report[key]["f1-score"] * 1000)
                                    / 1000,
                                    "c_f1-score": int(c_report[key]["f1-score"] * 1000)
                                    / 1000,
                                    "support": report[key]["support"],
                                    "c_support": c_report[key]["support"],
                                }
                            )
                        else:
                            table_list.append(
                                {
                                    "class": key,
                                    "precision": int(report[key]["precision"] * 1000)
                                    / 1000,
                                    "c_precision": 0,
                                    "recall": int(report[key]["recall"] * 1000) / 1000,
                                    "c_recall": 0,
                                    "f1-score": int(report[key]["f1-score"] * 1000)
                                    / 1000,
                                    "c_f1-score": 0,
                                    "support": report[key]["support"],
                                    "c_support": 0,
                                }
                            )
                ctx.panel.set_data("my_stack.c_evaluations", table_list)

            # Do the same behaviour of checking classes in detection as well
            elif eval_type == "detection":
                for key in report:
                    if key in c_keys:
                        table_list.append(
                            {
                                "class": key,
                                "precision": int(report[key]["precision"] * 1000)
                                / 1000,
                                "c_precision": int(c_report[key]["precision"] * 1000)
                                / 1000,
                                "recall": int(report[key]["recall"] * 1000) / 1000,
                                "c_recall": int(c_report[key]["recall"] * 1000) / 1000,
                                "f1-score": int(report[key]["f1-score"] * 1000) / 1000,
                                "c_f1-score": int(c_report[key]["f1-score"] * 1000)
                                / 1000,
                                "support": report[key]["support"],
                                "c_support": c_report[key]["support"],
                            }
                        )
                    else:
                        table_list.append(
                            {
                                "class": key,
                                "precision": int(report[key]["precision"] * 1000)
                                / 1000,
                                "c_precision": 0,
                                "recall": int(report[key]["recall"] * 1000) / 1000,
                                "c_recall": 0,
                                "f1-score": int(report[key]["f1-score"] * 1000) / 1000,
                                "c_f1-score": 0,
                                "support": report[key]["support"],
                                "c_support": 0,
                            }
                        )
                ctx.panel.set_data("my_stack.c_evaluations", table_list)
                if info["config"]["compute_mAP"]:
                    if c_eval["config"]["compute_mAP"]:
                        mAP_list = []
                        for i, label in enumerate(results.classes):
                            if label in c_keys:
                                c_label_index = c_keys.index(label)
                                new_row = {
                                    "class": label,
                                    "AP": int(results._classwise_AP[i] * 1000) / 1000,
                                    "c_AP": int(
                                        c_results._classwise_AP[c_label_index] * 1000
                                    )
                                    / 1000,
                                }
                                mAP_list.append(new_row)
                            else:
                                new_row = {
                                    "class": label,
                                    "AP": int(results._classwise_AP[i] * 1000) / 1000,
                                    "c_AP": 0,
                                }
                                mAP_list.append(new_row)

                        new_row = {
                            "class": "All",
                            "AP": int(results.mAP() * 1000) / 1000,
                            "c_AP": int(c_results.mAP() * 1000) / 1000,
                        }
                        mAP_list.append(new_row)

                    else:
                        mAP_list = []
                        for i, label in enumerate(results.classes):
                            new_row = {
                                "class": label,
                                "AP": int(results._classwise_AP[i] * 1000) / 1000,
                                "c_AP": 0,
                            }
                            mAP_list.append(new_row)

                        new_row = {
                            "class": "All",
                            "AP": int(results.mAP() * 1000) / 1000,
                            "c_AP": 0,
                        }
                        mAP_list.append(new_row)
                    ctx.panel.set_data("my_stack.c_mAP_evaluations", mAP_list)

    def _update_plot_data(
        self,
        ctx,
    ):
        # _update_plot_data is called in on_change_config
        # The function updates the DATA of all the plots in the panel,
        # including histograms and confusion matrices# _update_plot_data is called in on_change_config
        # The function updates the DATA of all the plots in the panel,
        # including histograms and confusion matrices

        # Grab the basic info
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
        path = ctx.params.get("path")
        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        results = ctx.dataset.load_evaluation_results(eval_key)
        pred_field = info["config"]["pred_field"]
        gt_field = info["config"]["gt_field"]
        eval_type = info["config"]["type"]

        # If only one eval is selected:
        if compare_key is None:

            # For detection evals, we need to use the nested field to get most of our info
            # Like f"{pred_field}.detections.field"
            if eval_type == "detection":

                # We will need classes for many of our histograms so load first
                classes = ctx.dataset.distinct(f"{pred_field}.detections.label")

                """
                Numerical Histograms are defined as follows:

                # Must be a numerical based field
                x = f"{pred_field}.detections.field"

                # bins are unconfigurable at the moment and hard coded to 10

                bins = 10

                # Next we compute the histogram with either fiftyones built in function or our own
                # This will retrun the counts for each bin, the edges of the bins. The other value can be ignored
                # we use those to define the the histograme which is given below. Name the trace, set the x and y in plotly
                # and set the data
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )
                counts = np.asarray(counts)
                edges = np.asarray(edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.detections.confidence",
                    "x": left_edges.tolist(),
                    "y": counts.tolist(),
                    "ids": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                    "type": "bar",
                    "width": widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.confidence",
                    [
                        # trace
                        histogram_data,
                    ],
                )
                """
                x = f"{pred_field}.detections.confidence"
                bins = 10
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )
                counts = np.asarray(counts)
                edges = np.asarray(edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.detections.confidence",
                    "x": left_edges.tolist(),
                    "y": counts.tolist(),
                    "ids": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                    "type": "bar",
                    "width": widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.confidence",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                x = f"{pred_field}.detections.{eval_key}_iou"
                bins = 10
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )
                counts = np.asarray(counts)
                edges = np.asarray(edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}_iou",
                    "x": left_edges.tolist(),
                    "y": counts.tolist(),
                    "type": "bar",
                    "width": widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.iou",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                # Calculate recall, precision, and f1. Dont forget to check for divide by 0!
                tp = np.array(ctx.dataset.values(f"{eval_key}_tp"))
                fp = np.array(ctx.dataset.values(f"{eval_key}_fp"))
                fn = np.array(ctx.dataset.values(f"{eval_key}_fn"))

                n = tp.astype(np.float64)
                d = (tp + fp).astype(np.float64)
                p = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                p = np.nan_to_num(p, nan=0.0)

                n = tp.astype(np.float64)
                d = (tp + fn).astype(np.float64)
                r = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                r = np.nan_to_num(r, nan=0.0)

                n = (2 * (p * r)).astype(np.float64)
                d = (p + r).astype(np.float64)
                f1 = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                f1 = np.nan_to_num(f1, nan=0.0)

                p_left_edges, p_counts, p_widths = compute_histogram(p, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": p_left_edges.tolist(),
                    "y": p_counts.tolist(),
                    "type": "bar",
                    "width": p_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.precision",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                r_left_edges, r_counts, r_widths = compute_histogram(r, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": r_left_edges.tolist(),
                    "y": r_counts.tolist(),
                    "type": "bar",
                    "width": r_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.recall",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                f1_left_edges, f1_counts, f1_widths = compute_histogram(f1, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": f1_left_edges.tolist(),
                    "y": f1_counts.tolist(),
                    "type": "bar",
                    "width": f1_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.f1",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                # Here we do it by class. We sum up the list of lists of each tp/fp/fn
                p_class_list = []
                r_class_list = []
                f1_class_list = []
                conf_class_list = []
                for cls in classes:
                    tp = sum(
                        sublist.count("tp")
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.{eval_key}")
                    )
                    fp = sum(
                        sublist.count("fp")
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.{eval_key}")
                    )
                    fn = sum(
                        sublist.count("fn")
                        for sublist in ctx.dataset.filter_labels(
                            gt_field, F("label").is_in([cls])
                        ).values(f"{gt_field}.detections.{eval_key}")
                    )

                    conf_total = [
                        item
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.confidence")
                        for item in sublist
                    ]

                    conf = sum(conf_total) / len(conf_total)

                    if tp + fp != 0:
                        p = tp / (tp + fp)
                        p = np.nan_to_num(p, nan=0.0)
                    else:
                        p = 0
                    if tp + fn != 0:
                        r = tp / (tp + fn)
                        r = np.nan_to_num(r, nan=0.0)
                    else:
                        r = 0
                    if p + r != 0:
                        f1 = 2 * (p * r) / (p + r)
                        f1 = np.nan_to_num(f1, nan=0.0)
                    else:
                        f1 = 0

                    p_class_list.append(p)
                    r_class_list.append(r)
                    f1_class_list.append(f1)
                    conf_class_list.append(conf)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": conf_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.conf_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )
                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": p_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.p_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": r_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.r_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": f1_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.f1_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                # We use FO builtin confusion_matrix func to create the matrix
                confusion_matrix, labels, ids = results._confusion_matrix(
                    include_other=False,
                    include_missing=True,
                )
                # confusion_matrix = list(map(list, zip(confusion_matrix)))

                cm_data = {
                    "z": confusion_matrix,
                    "x": labels,
                    "y": labels,
                    "type": "heatmap",
                }
                ctx.panel.set_data(
                    "my_stack.cm",
                    [
                        # trace
                        cm_data,
                    ],
                )
            elif eval_type == "classification":
                classes = ctx.dataset.distinct(f"{pred_field}.label")

                x = f"{pred_field}.confidence"
                bins = 10
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )
                counts = np.asarray(counts)
                edges = np.asarray(edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.confidence",
                    "x": left_edges.tolist(),
                    "ids": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                    "y": counts.tolist(),
                    "type": "bar",
                    "width": widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.confidence",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                p_class_list = []
                r_class_list = []
                f1_class_list = []
                conf_class_list = []
                for cls in classes:
                    tp = (
                        ctx.dataset.filter_labels(pred_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(True)
                    )
                    fp = (
                        ctx.dataset.filter_labels(pred_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(False)
                    )
                    fn = (
                        ctx.dataset.filter_labels(gt_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(False)
                    )
                    conf_total = ctx.dataset.filter_labels(
                        pred_field, F("label").is_in([cls])
                    ).values(f"{pred_field}.confidence")

                    conf = sum(conf_total) / len(conf_total)

                    if tp + fp != 0:
                        p = tp / (tp + fp)
                    else:
                        p = 0
                    p = np.nan_to_num(p, nan=0.0)
                    if tp + fn != 0:
                        r = tp / (tp + fn)
                    else:
                        r = 0
                    r = np.nan_to_num(r, nan=0.0)
                    if p + r != 0:
                        f1 = 2 * (p * r) / (p + r)
                    else:
                        f1 = 0
                    f1 = np.nan_to_num(f1, nan=0.0)

                    p_class_list.append(p)
                    r_class_list.append(r)
                    f1_class_list.append(f1)
                    conf_class_list.append(conf)

                histogram_data = {
                    "name": f"{pred_field}.confidence",
                    "x": classes,
                    "y": conf_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.conf_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )
                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": p_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.p_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": r_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.r_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": f1_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.f1_class",
                    [
                        # trace
                        histogram_data,
                    ],
                )

                confusion_matrix, labels, ids = results._confusion_matrix(
                    include_other=False,
                    include_missing=False,
                )
                # confusion_matrix = list(map(list, zip(confusion_matrix)))

                cm_data = {
                    "z": confusion_matrix,
                    "x": labels,
                    "y": labels,
                    "type": "heatmap",
                }
                ctx.panel.set_data(
                    "my_stack.cm",
                    [
                        # trace
                        cm_data,
                    ],
                )
        else:
            c_info = ctx.dataset.get_evaluation_info(compare_key).serialize()
            c_results = ctx.dataset.load_evaluation_results(compare_key)
            c_pred_field = c_info["config"]["pred_field"]
            c_gt_field = c_info["config"]["gt_field"]

            if eval_type == "detection":
                classes = ctx.dataset.distinct(f"{pred_field}.detections.label")
                c_classes = ctx.dataset.distinct(f"{c_pred_field}.detections.label")

                bins = 10
                x = f"{pred_field}.detections.confidence"
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )

                x = f"{c_pred_field}.detections.confidence"
                c_counts, c_edges, c_other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )

                counts = np.asarray(counts)
                edges = np.asarray(edges)
                c_counts = np.asarray(c_counts)
                c_edges = np.asarray(c_edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                c_left_edges = c_edges[:-1]
                c_widths = c_edges[1:] - c_edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.detections.confidence",
                    "x": left_edges.tolist(),
                    "y": counts.tolist(),
                    "width": widths.tolist(),
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.confidence",
                    "x": c_left_edges.tolist(),
                    "y": c_counts.tolist(),
                    "width": c_widths.tolist(),
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.confidence",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                bins = 10
                x = f"{pred_field}.detections.{eval_key}_iou"
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )

                x = f"{c_pred_field}.detections.{compare_key}_iou"
                c_counts, c_edges, c_other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )

                counts = np.asarray(counts)
                edges = np.asarray(edges)
                c_counts = np.asarray(c_counts)
                c_edges = np.asarray(c_edges)

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                c_left_edges = c_edges[:-1]
                c_widths = c_edges[1:] - c_edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}_iou",
                    "x": left_edges.tolist(),
                    "y": counts.tolist(),
                    "width": widths.tolist(),
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}_iou",
                    "x": c_left_edges.tolist(),
                    "y": c_counts.tolist(),
                    "width": c_widths.tolist(),
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.iou",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                tp = np.array(ctx.dataset.values(f"{eval_key}_tp"))
                fp = np.array(ctx.dataset.values(f"{eval_key}_fp"))
                fn = np.array(ctx.dataset.values(f"{eval_key}_fn"))

                c_tp = np.array(ctx.dataset.values(f"{compare_key}_tp"))
                c_fp = np.array(ctx.dataset.values(f"{compare_key}_fp"))
                c_fn = np.array(ctx.dataset.values(f"{compare_key}_fn"))

                n = tp.astype(np.float64)
                d = (tp + fp).astype(np.float64)
                p = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                p = np.nan_to_num(p, nan=0.0)

                n = tp.astype(np.float64)
                d = (tp + fn).astype(np.float64)
                r = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                r = np.nan_to_num(r, nan=0.0)

                n = (2 * (p * r)).astype(np.float64)
                d = (p + r).astype(np.float64)
                f1 = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                f1 = np.nan_to_num(f1, nan=0.0)

                n = c_tp.astype(np.float64)
                d = (c_tp + c_fp).astype(np.float64)
                c_p = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                c_p = np.nan_to_num(c_p, nan=0.0)

                n = c_tp.astype(np.float64)
                d = (c_tp + c_fn).astype(np.float64)
                c_r = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                c_r = np.nan_to_num(r, nan=0.0)

                n = (2 * (c_p * c_r)).astype(np.float64)
                d = (c_p + c_r).astype(np.float64)
                c_f1 = np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)
                c_f1 = np.nan_to_num(f1, nan=0.0)

                p_left_edges, p_counts, p_widths = compute_histogram(p, 10)
                c_p_left_edges, c_p_counts, c_p_widths = compute_histogram(c_p, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": p_left_edges.tolist(),
                    "y": p_counts.tolist(),
                    "type": "bar",
                    "width": p_widths.tolist(),
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": c_p_left_edges.tolist(),
                    "y": c_p_counts.tolist(),
                    "type": "bar",
                    "width": c_p_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.precision",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                r_left_edges, r_counts, r_widths = compute_histogram(r, 10)
                c_r_left_edges, c_r_counts, c_r_widths = compute_histogram(c_r, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": r_left_edges.tolist(),
                    "y": r_counts.tolist(),
                    "type": "bar",
                    "width": r_widths.tolist(),
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": c_r_left_edges.tolist(),
                    "y": c_r_counts.tolist(),
                    "type": "bar",
                    "width": c_r_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.recall",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                f1_left_edges, f1_counts, f1_widths = compute_histogram(f1, 10)
                c_f1_left_edges, c_f1_counts, c_f1_widths = compute_histogram(c_f1, 10)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": f1_left_edges.tolist(),
                    "y": f1_counts.tolist(),
                    "type": "bar",
                    "width": f1_widths.tolist(),
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": c_f1_left_edges.tolist(),
                    "y": c_f1_counts.tolist(),
                    "type": "bar",
                    "width": c_f1_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.f1",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                p_class_list = []
                r_class_list = []
                f1_class_list = []
                conf_class_list = []

                c_p_class_list = []
                c_r_class_list = []
                c_f1_class_list = []
                c_conf_class_list = []

                for cls in classes:
                    tp = sum(
                        sublist.count("tp")
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.{eval_key}")
                    )
                    fp = sum(
                        sublist.count("fp")
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.{eval_key}")
                    )
                    fn = sum(
                        sublist.count("fn")
                        for sublist in ctx.dataset.filter_labels(
                            gt_field, F("label").is_in([cls])
                        ).values(f"{gt_field}.detections.{eval_key}")
                    )

                    conf_total = [
                        item
                        for sublist in ctx.dataset.filter_labels(
                            pred_field, F("label").is_in([cls])
                        ).values(f"{pred_field}.detections.confidence")
                        for item in sublist
                    ]

                    conf = sum(conf_total) / len(conf_total)

                    if tp + fp != 0:
                        p = tp / (tp + fp)
                        p = np.nan_to_num(p, nan=0.0)
                    else:
                        p = 0
                    if tp + fn != 0:
                        r = tp / (tp + fn)
                        r = np.nan_to_num(r, nan=0.0)
                    else:
                        r = 0
                    if p + r != 0:
                        f1 = 2 * (p * r) / (p + r)
                        f1 = np.nan_to_num(f1, nan=0.0)
                    else:
                        f1 = 0

                    p_class_list.append(p)
                    r_class_list.append(r)
                    f1_class_list.append(f1)
                    conf_class_list.append(conf)

                    if cls not in c_classes:
                        p_class_list.append(None)
                        r_class_list.append(None)
                        f1_class_list.append(None)
                        conf_class_list.append(None)
                    else:
                        c_tp = sum(
                            sublist.count("tp")
                            for sublist in ctx.dataset.filter_labels(
                                c_pred_field, F("label").is_in([cls])
                            ).values(f"{c_pred_field}.detections.{compare_key}")
                        )
                        c_fp = sum(
                            sublist.count("fp")
                            for sublist in ctx.dataset.filter_labels(
                                c_pred_field, F("label").is_in([cls])
                            ).values(f"{c_pred_field}.detections.{compare_key}")
                        )
                        c_fn = sum(
                            sublist.count("fn")
                            for sublist in ctx.dataset.filter_labels(
                                c_gt_field, F("label").is_in([cls])
                            ).values(f"{c_gt_field}.detections.{compare_key}")
                        )

                        c_conf_total = [
                            item
                            for sublist in ctx.dataset.filter_labels(
                                c_pred_field, F("label").is_in([cls])
                            ).values(f"{c_pred_field}.detections.confidence")
                            for item in sublist
                        ]

                        c_conf = sum(c_conf_total) / len(c_conf_total)

                        if c_tp + c_fp != 0:
                            c_p = c_tp / (c_tp + c_fp)
                            c_p = np.nan_to_num(c_p, nan=0.0)
                        else:
                            c_p = 0
                        if c_tp + c_fn != 0:
                            c_r = c_tp / (c_tp + c_fn)
                            c_r = np.nan_to_num(c_r, nan=0.0)
                        else:
                            c_r = 0
                        if c_p + c_r != 0:
                            c_f1 = 2 * (c_p * c_r) / (c_p + c_r)
                            c_f1 = np.nan_to_num(c_f1, nan=0.0)
                        else:
                            c_f1 = 0

                        c_p_class_list.append(c_p)
                        c_r_class_list.append(c_r)
                        c_f1_class_list.append(c_f1)
                        c_conf_class_list.append(c_conf)

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": conf_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": classes,
                    "y": c_conf_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.conf_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )
                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": p_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{gt_field}.detections.{compare_key}",
                    "x": classes,
                    "y": c_p_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.p_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": r_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": classes,
                    "y": c_r_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.r_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.detections.{eval_key}",
                    "x": classes,
                    "y": f1_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.detections.{compare_key}",
                    "x": classes,
                    "y": c_f1_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.f1_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )
                confusion_matrix, labels, ids = results._confusion_matrix(
                    include_other=False,
                    include_missing=True,
                )
                # confusion_matrix = list(map(list, zip(confusion_matrix)))

                cm_data = {
                    "z": confusion_matrix,
                    "x": labels,
                    "y": labels,
                    "type": "heatmap",
                }

                c_confusion_matrix, c_labels, c_ids = c_results._confusion_matrix(
                    include_other=False,
                    include_missing=True,
                )
                # c_confusion_matrix = list(map(list, zip(c_confusion_matrix)))

                c_cm_data = {
                    "z": c_confusion_matrix,
                    "x": c_labels,
                    "y": c_labels,
                    "type": "heatmap",
                }
                ctx.panel.set_data(
                    "my_stack.cm",
                    [
                        # trace
                        cm_data,
                    ],
                )
                ctx.panel.set_data(
                    "my_stack.c_cm",
                    [
                        # trace
                        c_cm_data,
                    ],
                )
            elif eval_type == "classification":
                classes = ctx.dataset.distinct(f"{pred_field}.label")
                c_classes = ctx.dataset.distinct(f"{c_pred_field}.label")

                x = f"{pred_field}.confidence"
                bins = 10
                counts, edges, other = ctx.dataset.histogram_values(
                    x,
                    bins=bins,
                )
                counts = np.asarray(counts)
                edges = np.asarray(edges)

                x = f"{c_pred_field}.confidence"

                left_edges = edges[:-1]
                widths = edges[1:] - edges[:-1]
                histogram_data = {
                    "name": f"{pred_field}.confidence",
                    "x": left_edges.tolist(),
                    "ids": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                    "y": counts.tolist(),
                    "type": "bar",
                    "width": widths.tolist(),
                }

                c_x = f"{c_pred_field}.confidence"
                c_bins = 10
                c_counts, c_edges, c_other = ctx.dataset.histogram_values(
                    c_x,
                    bins=c_bins,
                )
                c_counts = np.asarray(c_counts)
                c_edges = np.asarray(c_edges)

                c_left_edges = c_edges[:-1]
                c_widths = c_edges[1:] - c_edges[:-1]
                c_histogram_data = {
                    "name": f"{c_pred_field}.confidence",
                    "x": c_left_edges.tolist(),
                    "ids": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
                    "y": c_counts.tolist(),
                    "type": "bar",
                    "width": c_widths.tolist(),
                }
                ctx.panel.set_data(
                    "my_stack.confidence",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                p_class_list = []
                r_class_list = []
                f1_class_list = []
                conf_class_list = []

                c_p_class_list = []
                c_r_class_list = []
                c_f1_class_list = []
                c_conf_class_list = []
                for cls in classes:
                    tp = (
                        ctx.dataset.filter_labels(pred_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(True)
                    )
                    fp = (
                        ctx.dataset.filter_labels(pred_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(False)
                    )
                    fn = (
                        ctx.dataset.filter_labels(gt_field, F("label").is_in([cls]))
                        .values(f"{eval_key}")
                        .count(False)
                    )
                    conf_total = ctx.dataset.filter_labels(
                        pred_field, F("label").is_in([cls])
                    ).values(f"{pred_field}.confidence")

                    conf = sum(conf_total) / len(conf_total)

                    if tp + fp != 0:
                        p = tp / (tp + fp)
                    else:
                        p = 0
                    p = np.nan_to_num(p, nan=0.0)
                    if tp + fn != 0:
                        r = tp / (tp + fn)
                    else:
                        r = 0
                    r = np.nan_to_num(r, nan=0.0)
                    if p + r != 0:
                        f1 = 2 * (p * r) / (p + r)
                    else:
                        f1 = 0
                    f1 = np.nan_to_num(f1, nan=0.0)

                    p_class_list.append(p)
                    r_class_list.append(r)
                    f1_class_list.append(f1)
                    conf_class_list.append(conf)

                    if cls not in c_classes:
                        p_class_list.append(None)
                        r_class_list.append(None)
                        f1_class_list.append(None)
                        conf_class_list.append(None)
                    else:
                        c_tp = (
                            ctx.dataset.filter_labels(
                                c_pred_field, F("label").is_in([cls])
                            )
                            .values(f"{compare_key}")
                            .count(True)
                        )
                        c_fp = (
                            ctx.dataset.filter_labels(
                                c_pred_field, F("label").is_in([cls])
                            )
                            .values(f"{compare_key}")
                            .count(False)
                        )
                        c_fn = (
                            ctx.dataset.filter_labels(gt_field, F("label").is_in([cls]))
                            .values(f"{compare_key}")
                            .count(False)
                        )
                        c_conf_total = ctx.dataset.filter_labels(
                            c_pred_field, F("label").is_in([cls])
                        ).values(f"{c_pred_field}.confidence")

                        c_conf = sum(c_conf_total) / len(c_conf_total)

                        c_p = c_tp / (c_tp + c_fp)
                        c_p = np.nan_to_num(c_p, nan=0.0)
                        c_r = c_tp / (c_tp + c_fn)
                        c_r = np.nan_to_num(c_r, nan=0.0)
                        c_f1 = 2 * (c_p * c_r) / (c_p + c_r)
                        c_f1 = np.nan_to_num(c_f1, nan=0.0)

                        c_p_class_list.append(c_p)
                        c_r_class_list.append(c_r)
                        c_f1_class_list.append(c_f1)
                        c_conf_class_list.append(c_conf)

                histogram_data = {
                    "name": f"{pred_field}.confidence",
                    "x": classes,
                    "y": conf_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.confidence",
                    "x": c_classes,
                    "y": c_conf_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.conf_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )
                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": p_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.{compare_key}",
                    "x": classes,
                    "y": c_p_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.p_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": r_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.{compare_key}",
                    "x": classes,
                    "y": c_r_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.r_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                histogram_data = {
                    "name": f"{pred_field}.{eval_key}",
                    "x": classes,
                    "y": f1_class_list,
                    "type": "bar",
                }
                c_histogram_data = {
                    "name": f"{c_pred_field}.{compare_key}",
                    "x": classes,
                    "y": c_f1_class_list,
                    "type": "bar",
                }
                ctx.panel.set_data(
                    "my_stack.f1_class",
                    [
                        # trace
                        histogram_data,
                        c_histogram_data,
                    ],
                )

                confusion_matrix, labels, ids = results._confusion_matrix(
                    include_other=False,
                    include_missing=False,
                )
                # confusion_matrix = list(map(list, zip(confusion_matrix)))

                cm_data = {
                    "z": confusion_matrix,
                    "x": labels,
                    "y": labels,
                    "type": "heatmap",
                }

                c_confusion_matrix, c_labels, c_ids = c_results._confusion_matrix(
                    include_other=False,
                    include_missing=False,
                )
                # c_confusion_matrix = list(map(list, zip(c_confusion_matrix)))

                c_cm_data = {
                    "z": c_confusion_matrix,
                    "x": c_labels,
                    "y": c_labels,
                    "type": "heatmap",
                }
                ctx.panel.set_data(
                    "my_stack.cm",
                    [
                        # trace
                        cm_data,
                    ],
                )
                ctx.panel.set_data(
                    "my_stack.c_cm",
                    [
                        # trace
                        c_cm_data,
                    ],
                )

    def _add_plots(self, ctx, stack):
        # This function adds the plots into the render function
        # Start the ones that always appear regardless of model type and follow by eval_type
        # specific ones afterwards

        # Start by grabbing some basics
        eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)

        info = ctx.dataset.get_evaluation_info(eval_key).serialize()
        eval_type = info["config"]["type"]

        # Next for each plot, we set the config and layout for UI elements
        # We remove scrollZoom so that when you scroll on the panel it scrolls down and doesn't
        # accidently zoom on the plot
        config = {"scrollZoom": False}
        layout = {
            "title": "Confidence",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {"title": "Confidence"},
            "yaxis": {"title": "Count"},
            "showlegend": True,
            "legend": {"x": 0, "y": 1, "showlegend": True},
        }

        # After the plot layout/config is defined, add the property to the stack with the
        # appropriate on_call and on_selected calls
        # TODO add on_selected
        stack.add_property(
            "confidence",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config,
                    layout=layout,
                    on_click=self.on_numerical_click,
                    on_selected=self.on_numerical_click,
                ),
            ),
        )

        if eval_type == "detection":
            config = {"scrollZoom": False}
            layout = {
                "title": "IOU",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {"title": "IOU"},
                "yaxis": {"title": "Count"},
                "showlegend": True,
                "legend": {"x": 0, "y": 1, "showlegend": True},
            }
            stack.add_property(
                "iou",
                types.Property(
                    types.List(types.Object()),
                    view=types.PlotlyView(
                        config=config, layout=layout, on_click=self.on_numerical_click
                    ),
                ),
            )

            config = {"scrollZoom": False}
            layout = {
                "title": "Precision Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {"title": "Precision per Sample"},
                "yaxis": {"title": "Count"},
                "showlegend": True,
                "legend": {"x": 0, "y": 1, "showlegend": True},
            }
            stack.add_property(
                "precision",
                types.Property(
                    types.List(types.Object()),
                    view=types.PlotlyView(
                        config=config, layout=layout, on_click=self.on_numerical_click
                    ),
                ),
            )

            config = {"scrollZoom": False}
            layout = {
                "title": "Recall Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {"title": "Recall per Sample"},
                "yaxis": {"title": "Count"},
                "showlegend": True,
                "legend": {"x": 0, "y": 1, "showlegend": True},
            }
            stack.add_property(
                "recall",
                types.Property(
                    types.List(types.Object()),
                    view=types.PlotlyView(
                        config=config, layout=layout, on_click=self.on_numerical_click
                    ),
                ),
            )

            config = {"scrollZoom": False}
            layout = {
                "title": "F1-Score Distribution",
                "bargap": 0,
                "bargroupgap": 0,
                "xaxis": {"title": "F1-Score per Sample"},
                "yaxis": {"title": "Count"},
                "showlegend": True,
                "legend": {"x": 0, "y": 1, "showlegend": True},
            }
            stack.add_property(
                "f1",
                types.Property(
                    types.List(types.Object()),
                    view=types.PlotlyView(
                        config=config, layout=layout, on_click=self.on_numerical_click
                    ),
                ),
            )

        config = {"scrollZoom": False}
        layout = {
            "title": "Confidence per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {"title": "Class"},
            "yaxis": {"title": "Confidence"},
            "showlegend": True,
            "legend": {"x": 0, "y": 1, "showlegend": True},
        }
        stack.add_property(
            "conf_class",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config, layout=layout, on_click=self.on_categorical_click
                ),
            ),
        )
        config = {"scrollZoom": False}
        layout = {
            "title": "Precision per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {"title": "Class"},
            "yaxis": {"title": "Precision"},
            "showlegend": True,
            "legend": {"x": 0, "y": 1, "showlegend": True},
        }
        stack.add_property(
            "p_class",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config, layout=layout, on_click=self.on_categorical_click
                ),
            ),
        )

        config = {"scrollZoom": False}
        layout = {
            "title": "Recall per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {"title": "Class"},
            "yaxis": {"title": "Recall"},
            "showlegend": True,
            "legend": {"x": 0, "y": 1, "showlegend": True},
        }
        stack.add_property(
            "r_class",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config, layout=layout, on_click=self.on_categorical_click
                ),
            ),
        )

        config = {"scrollZoom": False}
        layout = {
            "title": "F1-Score per Class",
            "bargap": 0,
            "bargroupgap": 0,
            "xaxis": {"title": "Class"},
            "yaxis": {"title": "F1-Score"},
            "showlegend": True,
            "legend": {"x": 0, "y": 1, "showlegend": True},
        }
        stack.add_property(
            "f1_class",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config, layout=layout, on_click=self.on_categorical_click
                ),
            ),
        )

        config = {}
        layout = {
            "title": f"Confusion Matrix for {eval_key}",
            "yaxis": {"fixedrange": True, "title": "Ground truth"},
            "xaxis": {"fixedrange": True, "title": "Model predictions"},
        }

        stack.add_property(
            "cm",
            types.Property(
                types.List(types.Object()),
                view=types.PlotlyView(
                    config=config, layout=layout, on_click=self.on_cm_click
                ),
            ),
        )

        compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
        if compare_key is not None:
            config = {}
            layout = {
                "title": f"Confusion Matrix for {compare_key}",
                "yaxis": {"fixedrange": True, "title": "Ground truth"},
                "xaxis": {"fixedrange": True, "title": "Model predictions"},
            }

            stack.add_property(
                "c_cm",
                types.Property(
                    types.List(types.Object()),
                    view=types.PlotlyView(
                        config=config, layout=layout, on_click=self.on_cm_click
                    ),
                ),
            )


def _eval_results(
    ctx,
    stack,
):
    # Adds the tables to the panel

    # Start by grabbing our basics
    eval_key = ctx.panel.get_state("my_stack.menu.actions.eval_key")
    compare_key = ctx.panel.get_state("my_stack.menu.actions.compare_key", None)
    path = ctx.params.get("path")
    info = ctx.dataset.get_evaluation_info(eval_key).serialize()
    results = ctx.dataset.load_evaluation_results(eval_key)
    pred_field = info["config"]["pred_field"]
    gt_field = info["config"]["gt_field"]
    eval_type = info["config"]["type"]
    eval_type = info["config"]["type"]

    # Adds a title if there are results
    stack.md("### Evaluation Results")

    # Single eval first
    if compare_key is None:
        # Define table structure
        item_obj = types.Object()
        item_obj.str("class")
        item_obj.float("precision")
        item_obj.float("recall")
        item_obj.float("f1-score")
        item_obj.float("support")

        # Define the table
        table_view = types.TableView()
        table_view.add_column("class", label="Class")
        table_view.add_column("precision", label="Precision")
        table_view.add_column("recall", label="Recall")
        table_view.add_column("f1-score", label="F1-Score")
        table_view.add_column("support", label="Support")

        # Add table to the stack
        stack.list("evaluations", element_type=item_obj, view=table_view)

        if eval_type and eval_type == "detection":
            if info["config"]["compute_mAP"]:
                mAP_obj = types.Object()
                mAP_obj.str("class")
                mAP_obj.float("AP")

                mAP_table_view = types.TableView()
                mAP_table_view.add_column("class", label="Class")
                mAP_table_view.add_column("AP", label="Average Precision")
                stack.list("mAP_evaluations", element_type=mAP_obj, view=mAP_table_view)
            else:
                stack.md("No mAP copmuted on this eval_key")
    else:
        # Define table structure
        item_obj = types.Object()
        item_obj.str("class")
        item_obj.float("precision")
        item_obj.float("c_precision")
        item_obj.float("recall")
        item_obj.float("c_recall")
        item_obj.float("f1-score")
        item_obj.float("c_f1-score")
        item_obj.float("support")
        item_obj.float("c_support")

        # Define the table
        table_view = types.TableView()
        table_view.add_column("class", label="Class")
        table_view.add_column("precision", label="Model 1 Precision")
        table_view.add_column("c_precision", label="Model 2 Precision")
        table_view.add_column("recall", label="Model 1 Recall")
        table_view.add_column("c_recall", label="Model 2 Recall")
        table_view.add_column("f1-score", label="Model 1 F1-Score")
        table_view.add_column("c_f1-score", label="Model 2 F1-Score")
        table_view.add_column("support", label="Model 1 Support")
        table_view.add_column("c_support", label="Model 2 Support")

        # Add table to the stack
        stack.list("c_evaluations", element_type=item_obj, view=table_view)

        if eval_type and eval_type == "detection":
            if info["config"]["compute_mAP"]:
                mAP_obj = types.Object()
                mAP_obj.str("class")
                mAP_obj.float("AP")
                mAP_obj.float("c_AP")

                mAP_table_view = types.TableView()
                mAP_table_view.add_column("class", label="Class")
                mAP_table_view.add_column("AP", label="Model 1Average Precision")
                mAP_table_view.add_column("c_AP", label="Model 2 Average Precision")
                stack.list(
                    "c_mAP_evaluations", element_type=mAP_obj, view=mAP_table_view
                )
            else:
                stack.md("No mAP copmuted on this eval_key")


def _eval_info(ctx, stack, info):

    # Shows the eval info in the second tab of the panel
    eval_info = info.serialize()

    if info is not None:
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


def register(p):
    p.register(EvaluationPanel)

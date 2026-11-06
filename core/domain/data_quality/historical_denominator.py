import pandas as pd


def build_historical_denominator(stock_registry, calendar, suspensions, quotes, start_date, end_date,
                                supported_exchanges=("SSE", "SZSE"), code_aliases=(), delist_date_exclusions=()):
    required_stock_columns = {"ts_code", "symbol", "exchange", "curr_type", "list_status", "list_date", "delist_date"}
    required_quote_columns = {"code", "trade_date", "open", "high", "low", "close"}
    if not required_stock_columns.issubset(stock_registry.columns):
        raise ValueError("证券身份快照缺少必要字段")
    if not {"cal_date"}.issubset(calendar.columns):
        raise ValueError("交易日历缺少 cal_date")
    if not {"ts_code", "trade_date", "suspend_timing"}.issubset(suspensions.columns):
        raise ValueError("停牌数据缺少必要字段")
    if not required_quote_columns.issubset(quotes.columns):
        raise ValueError("现库行情快照缺少必要字段")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    registry = stock_registry.copy()
    registry["list_date"] = pd.to_datetime(registry["list_date"], errors="coerce")
    registry["delist_date"] = pd.to_datetime(registry["delist_date"], errors="coerce")
    registry["symbol"] = registry["symbol"].astype(str).str.strip()
    registry["exchange"] = registry["exchange"].astype(str).str.upper()
    registry["curr_type"] = registry["curr_type"].astype(str).str.upper()
    registry["list_status"] = registry["list_status"].astype(str).str.upper()
    valid_code = registry["symbol"].str.fullmatch(r"\d{6}")
    supported_exchanges = {str(exchange).upper() for exchange in supported_exchanges}
    valid_exchange = registry["exchange"].isin(supported_exchanges)
    valid_currency = registry["curr_type"].eq("CNY")
    relevant = registry["list_date"].le(end) & (registry["delist_date"].isna() | registry["delist_date"].ge(start))
    valid_status = registry["list_status"].isin(("L", "D"))
    included = registry.loc[relevant & valid_code & valid_exchange & valid_currency & valid_status].copy()
    blockers = []
    in_scope = registry["exchange"].isin(supported_exchanges)
    unknown_scope = ~registry["exchange"].isin((*supported_exchanges, "BSE"))
    unsupported = registry.loc[relevant & (in_scope | unknown_scope) & ~(valid_code & valid_exchange & valid_currency & valid_status)]
    if not unsupported.empty:
        blockers.append(f"{len(unsupported)} 条相关证券身份记录不满足 A 股代码/交易所/币种/上市状态合同")
    if registry.loc[relevant, "list_date"].isna().any():
        blockers.append("相关证券身份记录存在缺失上市日期")
    if registry.loc[relevant & registry["list_status"].eq("D"), "delist_date"].isna().any():
        blockers.append("相关退市证券缺少退市日期")
    if included["symbol"].duplicated().any():
        raise ValueError("证券身份快照存在重复六位代码")

    dates = calendar[["cal_date"]].copy()
    dates["trade_date"] = pd.to_datetime(dates.pop("cal_date"), errors="coerce")
    dates = dates.loc[dates["trade_date"].between(start, end)]
    if dates.empty or dates["trade_date"].isna().any() or dates["trade_date"].duplicated().any():
        raise ValueError("交易日历为空、含无效日期或重复日期")

    stocks = included[["symbol", "ts_code", "exchange", "list_status", "list_date", "delist_date"]].rename(columns={"symbol": "code"})
    expected = stocks.merge(dates, how="cross")
    expected = expected.loc[
        expected["trade_date"].ge(expected["list_date"])
        & (expected["delist_date"].isna() | expected["trade_date"].le(expected["delist_date"]))
    ].copy()
    expected["code"] = expected["code"].astype(str).str.zfill(6)
    excluded_delist_dates = pd.DataFrame(delist_date_exclusions).copy()
    if not excluded_delist_dates.empty:
        required_exclusion_columns = {"code", "ts_code", "trade_date", "adjudication_id"}
        if not required_exclusion_columns.issubset(excluded_delist_dates.columns):
            raise ValueError("退市日排除裁决缺少必要字段")
        excluded_delist_dates["code"] = excluded_delist_dates["code"].astype(str).str.zfill(6)
        excluded_delist_dates["ts_code"] = excluded_delist_dates["ts_code"].astype(str).str.upper()
        excluded_delist_dates["trade_date"] = pd.to_datetime(excluded_delist_dates["trade_date"], errors="coerce")
        if (
            excluded_delist_dates["trade_date"].isna().any()
            or excluded_delist_dates.duplicated(["code", "trade_date"]).any()
        ):
            raise ValueError("退市日排除裁决日期无效或存在重复键")
        excluded_delist_dates = excluded_delist_dates.loc[
            excluded_delist_dates["trade_date"].between(start, end)
        ].copy()
        if not excluded_delist_dates.empty:
            registry_dates = stocks[["code", "ts_code", "delist_date"]].drop_duplicates()
            exclusion_identity = excluded_delist_dates.merge(
                registry_dates, on=["code", "ts_code"], how="left", validate="many_to_one",
            )
            if exclusion_identity["delist_date"].isna().any() or not exclusion_identity["trade_date"].eq(exclusion_identity["delist_date"]).all():
                raise ValueError("退市日排除裁决与证券清单退市日期不一致")
            expected = expected.merge(
                excluded_delist_dates[["code", "trade_date"]].assign(_exclude_delist_date=True),
                on=["code", "trade_date"], how="left", validate="one_to_one",
            )
            excluded_delist_dates = expected.loc[expected["_exclude_delist_date"].eq(True), [
                "code", "ts_code", "trade_date", "_exclude_delist_date",
            ]].merge(
                excluded_delist_dates[["code", "ts_code", "trade_date", "adjudication_id"]],
                on=["code", "ts_code", "trade_date"], how="left", validate="one_to_one",
            ).rename(columns={"_exclude_delist_date": "verified_delisted"})
            expected = expected.loc[~expected["_exclude_delist_date"].eq(True)].drop(columns="_exclude_delist_date")
        else:
            excluded_delist_dates = pd.DataFrame(columns=["code", "ts_code", "trade_date", "adjudication_id", "verified_delisted"])
    else:
        excluded_delist_dates = pd.DataFrame(columns=["code", "ts_code", "trade_date", "adjudication_id", "verified_delisted"])
    if expected.duplicated(["code", "trade_date"]).any():
        raise ValueError("构造的历史分母存在重复股票交易日")

    suspension_source_rows = suspensions[["ts_code", "trade_date", "suspend_timing"]].copy()
    suspension_source_rows["ts_code"] = suspension_source_rows["ts_code"].astype(str).str.upper()
    suspension_source_rows["trade_date"] = pd.to_datetime(suspension_source_rows["trade_date"], errors="coerce")
    invalid_suspension_dates = suspension_source_rows["trade_date"].isna()
    if invalid_suspension_dates.any():
        blockers.append(f"{int(invalid_suspension_dates.sum())} 条停牌源记录含无效日期")
    suspension_source_rows = suspension_source_rows.loc[
        ~invalid_suspension_dates & suspension_source_rows["trade_date"].between(start, end)
    ].copy()
    suffix_by_exchange = {"SSE": ".SH", "SZSE": ".SZ", "BSE": ".BJ"}
    target_suffixes = tuple(suffix_by_exchange[exchange] for exchange in supported_exchanges if exchange in suffix_by_exchange)
    excluded_suffixes = tuple(
        suffix for exchange, suffix in suffix_by_exchange.items() if exchange not in supported_exchanges
    )
    out_of_scope_suspensions = suspension_source_rows.loc[
        suspension_source_rows["ts_code"].str.endswith(excluded_suffixes)
    ].copy()
    unclassified_suspensions = suspension_source_rows.loc[
        ~suspension_source_rows["ts_code"].str.endswith((*target_suffixes, *excluded_suffixes))
    ].copy()
    suspension_source_rows = suspension_source_rows.loc[
        suspension_source_rows["ts_code"].str.endswith(target_suffixes)
    ].copy()
    if not unclassified_suspensions.empty:
        blockers.append(f"{len(unclassified_suspensions)} 条停牌源记录无法识别交易所范围")
    identity_lookup = registry.loc[in_scope, [
        "ts_code", "symbol", "exchange", "curr_type", "list_status", "list_date", "delist_date",
    ]].drop_duplicates("ts_code")
    suspension_identity = suspension_source_rows.merge(
        identity_lookup, on="ts_code", how="left", indicator=True, validate="many_to_one",
    )
    unmapped_suspensions = suspension_identity.loc[suspension_identity["_merge"] == "left_only"].copy()
    included_ts_codes = set(stocks["ts_code"])
    excluded_suspensions = suspension_identity.loc[
        (suspension_identity["_merge"] == "both") & ~suspension_identity["ts_code"].isin(included_ts_codes)
    ].copy()
    unmapped_suspensions = unmapped_suspensions.drop(columns="_merge")
    excluded_suspensions = excluded_suspensions.drop(columns="_merge")
    if not unmapped_suspensions.empty:
        blockers.append(f"{len(unmapped_suspensions)} 条停牌源记录缺少当前证券身份映射")
    if not excluded_suspensions.empty:
        blockers.append(f"{len(excluded_suspensions)} 条目标市场停牌源记录对应的证券未进入目标股票池")

    suspended = suspension_source_rows.copy()
    suspended["full_day_suspended"] = suspended["suspend_timing"].isna() | suspended["suspend_timing"].astype(str).str.strip().eq("")
    suspension_codes = stocks[["code", "ts_code", "list_date", "delist_date"]].drop_duplicates()
    suspended = suspended.merge(suspension_codes, on="ts_code", how="inner", validate="many_to_one")
    if suspended["trade_date"].isna().any() or suspended.duplicated(["code", "trade_date"]).any():
        raise ValueError("停牌数据含无效日期或重复股票交易日")
    calendar_days = set(dates["trade_date"])
    if not set(suspended["trade_date"]).issubset(calendar_days):
        blockers.append("停牌接口包含未出现在交易日历中的日期")
    suspension_expected = suspended[["code", "trade_date"]].merge(
        expected[["code", "trade_date"]], on=["code", "trade_date"], how="left", indicator=True,
    )
    suspension_outside_expected = suspension_expected.loc[suspension_expected["_merge"] == "left_only"]
    if not suspension_outside_expected.empty:
        blockers.append(f"{len(suspension_outside_expected)} 条停牌记录不在证券有效上市区间")
    suspension_outside_expected = suspension_outside_expected.merge(
        suspended[["code", "trade_date", "ts_code", "suspend_timing", "list_date", "delist_date"]],
        on=["code", "trade_date"], how="left", validate="one_to_one",
    ).drop(columns="_merge")

    quote_rows = quotes.copy()
    quote_rows["code"] = quote_rows["code"].astype(str).str.zfill(6)
    quote_rows["trade_date"] = pd.to_datetime(quote_rows["trade_date"], errors="coerce")
    quote_rows = quote_rows.loc[quote_rows["trade_date"].between(start, end)]
    out_of_scope_symbols = set(registry.loc[
        registry["exchange"].eq("BSE") & ~registry["exchange"].isin(supported_exchanges), "symbol"
    ].astype(str).str.zfill(6))
    quote_rows = quote_rows.loc[~quote_rows["code"].isin(out_of_scope_symbols)]
    legacy_alias_frames = []
    alias_symbols = set()
    for alias in code_aliases:
        old_symbol = str(alias.get("old_symbol", "")).zfill(6)
        canonical_symbol = str(alias.get("canonical_symbol", "")).zfill(6)
        effective_date = pd.to_datetime(alias.get("effective_date"), errors="coerce")
        old_ts_code = str(alias.get("old_ts_code", "")).upper()
        canonical_ts_code = str(alias.get("canonical_ts_code", "")).upper()
        if (not old_symbol.isdigit() or not canonical_symbol.isdigit() or pd.isna(effective_date)
                or not old_ts_code or not canonical_ts_code or old_symbol == canonical_symbol):
            blockers.append("证券代码别名清单存在无效映射")
            continue
        if not registry["ts_code"].astype(str).str.upper().eq(canonical_ts_code).any():
            blockers.append(f"证券代码别名 {old_ts_code} 的规范代码不在身份快照中")
            continue
        alias_symbols.add(old_symbol)
        legacy_rows = quote_rows.loc[
            quote_rows["code"].eq(old_symbol) & quote_rows["trade_date"].lt(effective_date)
        ].copy()
        if not legacy_rows.empty:
            legacy_rows["canonical_code"] = canonical_symbol
            legacy_rows["alias_effective_date"] = effective_date
            legacy_alias_frames.append(legacy_rows)
        quote_rows = quote_rows.loc[~(quote_rows["code"].eq(old_symbol) & quote_rows["trade_date"].lt(effective_date))]
    legacy_alias_quotes = pd.concat(legacy_alias_frames, ignore_index=True) if legacy_alias_frames else pd.DataFrame(
        columns=[*quote_rows.columns, "canonical_code", "alias_effective_date"]
    )
    legacy_alias_unverified = quote_rows.loc[quote_rows["code"].isin(alias_symbols)].copy()
    if not legacy_alias_unverified.empty:
        blockers.append(f"{len(legacy_alias_unverified)} 条旧证券代码在切换生效日后仍有行情记录")
    if quote_rows["trade_date"].isna().any() or quote_rows.duplicated(["code", "trade_date"]).any():
        raise ValueError("现库行情快照含无效日期或重复股票交易日")
    quote_rows["database_row_present"] = True
    quote_rows["database_ohlc_complete"] = quote_rows[["open", "high", "low", "close"]].notna().all(axis=1)
    prices = quote_rows[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    quote_rows["database_ohlc_valid"] = (
        prices.gt(0).all(axis=1)
        & prices["high"].ge(prices["low"])
        & prices["open"].between(prices["low"], prices["high"])
        & prices["close"].between(prices["low"], prices["high"])
    )
    expected = expected.merge(
        suspended[["code", "trade_date", "suspend_timing", "full_day_suspended"]],
        on=["code", "trade_date"], how="left", validate="one_to_one",
    )
    expected = expected.merge(
        quote_rows[["code", "trade_date", "database_row_present", "database_ohlc_complete", "database_ohlc_valid"]],
        on=["code", "trade_date"], how="left", validate="one_to_one",
    )
    quote_keys = quote_rows[[
        "code", "trade_date", "open", "high", "low", "close",
        "database_ohlc_complete", "database_ohlc_valid",
    ]].drop_duplicates()
    adjudicated_delist_quote_keys = quote_keys.iloc[0:0].copy()
    if not excluded_delist_dates.empty:
        adjudicated_delist_quote_keys = quote_keys.merge(
            excluded_delist_dates[["code", "trade_date"]],
            on=["code", "trade_date"], how="inner", validate="one_to_one",
        )
    unexpected_quote_keys = quote_keys.merge(
        expected[["code", "trade_date"]], on=["code", "trade_date"], how="left", indicator=True,
    )
    unexpected_quote_keys = unexpected_quote_keys.loc[unexpected_quote_keys["_merge"] == "left_only"]
    if not adjudicated_delist_quote_keys.empty:
        unexpected_quote_keys = unexpected_quote_keys.merge(
            adjudicated_delist_quote_keys[["code", "trade_date"]].assign(_adjudicated_delist_date=True),
            on=["code", "trade_date"], how="left", validate="one_to_one",
        )
        unexpected_quote_keys = unexpected_quote_keys.loc[
            ~unexpected_quote_keys["_adjudicated_delist_date"].eq(True)
        ].drop(columns="_adjudicated_delist_date")
    if not unexpected_quote_keys.empty:
        blockers.append(f"{len(unexpected_quote_keys)} 条现库行情股票日期不在身份快照的有效上市区间内")
    expected["full_day_suspended"] = expected["full_day_suspended"].eq(True)
    expected["database_row_present"] = expected["database_row_present"].eq(True)
    expected["database_ohlc_complete"] = expected["database_ohlc_complete"].eq(True)
    expected["database_ohlc_valid"] = expected["database_ohlc_valid"].eq(True)
    expected["partial_suspension"] = False
    expected["coverage_status"] = "missing_row_unresolved"
    expected.loc[expected["database_row_present"] & ~expected["database_ohlc_complete"], "coverage_status"] = "observed_row_missing_price"
    expected.loc[expected["database_row_present"] & expected["database_ohlc_complete"] & ~expected["database_ohlc_valid"], "coverage_status"] = "observed_row_invalid_price"
    expected.loc[expected["database_row_present"] & expected["database_ohlc_valid"], "coverage_status"] = "observed_price_complete"
    expected.loc[expected["full_day_suspended"], "coverage_status"] = "suspended"
    expected.loc[expected["suspend_timing"].notna() & ~expected["full_day_suspended"], "partial_suspension"] = True

    counts = expected["coverage_status"].value_counts().to_dict()
    summary = {
        "expected_stock_dates": int(len(expected)),
        "expected_trading_stock_dates": int(len(expected) - counts.get("suspended", 0)),
        "suspended_stock_dates": int(counts.get("suspended", 0)),
        "observed_price_complete": int(counts.get("observed_price_complete", 0)),
        "observed_row_missing_price": int(counts.get("observed_row_missing_price", 0)),
        "observed_row_invalid_price": int(counts.get("observed_row_invalid_price", 0)),
        "observed_invalid_price_rows": int((expected["database_row_present"] & expected["database_ohlc_complete"] & ~expected["database_ohlc_valid"]).sum()),
        "missing_row_unresolved": int(counts.get("missing_row_unresolved", 0)),
        "partial_suspension_stock_dates": int(expected["partial_suspension"].sum()),
        "observed_rows_outside_expected": int(len(unexpected_quote_keys)),
        "adjudicated_delist_quote_rows": int(len(adjudicated_delist_quote_keys)),
        "legacy_alias_quote_rows": int(len(legacy_alias_quotes)),
        "legacy_alias_unverified_rows": int(len(legacy_alias_unverified)),
        "unexpected_rows_with_invalid_ohlc": int((unexpected_quote_keys["database_ohlc_complete"] & ~unexpected_quote_keys["database_ohlc_valid"]).sum()),
        "suspension_rows_unmapped_identity": int(len(unmapped_suspensions)),
        "suspension_rows_known_but_excluded": int(len(excluded_suspensions)),
        "suspension_rows_outside_expected": int(len(suspension_outside_expected)),
        "suspension_rows_out_of_scope": int(len(out_of_scope_suspensions)),
        "suspension_rows_unclassified_scope": int(len(unclassified_suspensions)),
        "suspension_rows_invalid_date": int(invalid_suspension_dates.sum()),
        "verified_delist_date_exclusions": int(len(excluded_delist_dates)),
        "included_securities": int(stocks["code"].nunique()),
        "supported_exchanges": sorted(supported_exchanges),
        "calendar_trade_dates": int(len(dates)),
        "source_blockers": blockers,
        "denominator_verified": not blockers,
    }
    stock_year = expected.assign(year=expected["trade_date"].dt.year).groupby(["code", "year", "coverage_status"]).size().unstack(fill_value=0).reset_index()
    return {
        "expected_keys": expected,
        "stock_year_summary": stock_year,
        "unexpected_quote_keys": unexpected_quote_keys.drop(columns="_merge"),
        "adjudicated_delist_quote_keys": adjudicated_delist_quote_keys,
        "legacy_alias_quotes": legacy_alias_quotes,
        "legacy_alias_unverified_quotes": legacy_alias_unverified,
        "unmapped_suspensions": unmapped_suspensions,
        "excluded_suspensions": excluded_suspensions,
        "out_of_scope_suspensions": out_of_scope_suspensions,
        "unclassified_suspensions": unclassified_suspensions,
        "suspensions_outside_expected": suspension_outside_expected,
        "excluded_delist_dates": excluded_delist_dates,
        "summary": summary,
    }

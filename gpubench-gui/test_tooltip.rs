use iced::widget::{text, tooltip};
use iced::widget::tooltip::Position;
use iced::{Element, Theme};

pub fn test() -> Element<'static, ()> {
    tooltip(
        text("(?)"),
        text("description text"),
        Position::Top
    ).into()
}
